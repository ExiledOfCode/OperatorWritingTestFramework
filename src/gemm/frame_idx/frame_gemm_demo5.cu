#include <algorithm>
#include <chrono> // 用于计时
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "op_registry.hpp"
#include "op_runtime.hpp"
#include "utils.cuh"

// 参考 CPU 实现
// static void gemm_cpu(const float *A, const float *B, float *C, int M, int N, int K) {
//     for (int m = 0; m < M; m++) {
//         for (int n = 0; n < N; n++) {
//             float acc = 0.f;
//             for (int k = 0; k < K; k++) {
//                 acc += A[m * K + k] * B[k * N + n];
//             }
//             C[m * N + n] = acc;
//         }
//     }
// }
// shared memory:
// read:
// write:   MN
// global memory:
// read:     (M/bm)*(N/bn)个块，每个块(bm*K+K*bn)次访存，总计：KMN(1/bm+1/bn)次
// write:    MN
template <typename T>
struct MatrixView {
    T *ptr;
    int width;
    __device__ __forceinline__ T &operator()(int r, int c) { return ptr[r * width + c]; }
};

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

template <unsigned int M_NUM_PER_BLOCK, unsigned int N_NUM_PER_BLOCK, unsigned int K_NUM_PER_BLOCK, unsigned int NUM_PER_THREAD, unsigned int REG_PER_THREAD>
__global__ void frame_gemm_demo5(float *dA, float *dB, float *dC, int M, int N, int K) {

    MatrixView<float> A{dA, K};
    MatrixView<float> B{dB, N};
    MatrixView<float> C{dC, N};

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int block_base_x = (blockDim.x * NUM_PER_THREAD) * blockIdx.x;
    int block_base_y = blockDim.y * blockIdx.y;

    // int col = tx + block_base_x;
    int row = ty + block_base_y;

    // 二维索引的映射变换
    int block_idx = tx + blockDim.x * ty;

    int reg_tx = block_idx % (K_NUM_PER_BLOCK / REG_PER_THREAD);
    int reg_ty = block_idx / (K_NUM_PER_BLOCK / REG_PER_THREAD);

    __shared__ float A_shared[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float B_shared[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];

    float a_reg[REG_PER_THREAD] = {0.0f};
    float b_reg[REG_PER_THREAD] = {0.0f};
    float tmp[REG_PER_THREAD * REG_PER_THREAD] = {0.0f};

    for (int s = 0; s < K / K_NUM_PER_BLOCK; s++) {
        const int k_base = s * K_NUM_PER_BLOCK;
        FETCH_FLOAT4(A_shared[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(A(row, tx * NUM_PER_THREAD + k_base));
        FETCH_FLOAT4(B_shared[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(B(ty + k_base, block_base_x + NUM_PER_THREAD * tx));

        __syncthreads();

        for (int k = 0; k < K_NUM_PER_BLOCK; k++) {
            a_reg[0] = A_shared[reg_ty * 2][k];
            a_reg[1] = A_shared[reg_ty * 2 + 1][k];
            b_reg[0] = B_shared[k][reg_tx * 2];
            b_reg[1] = B_shared[k][reg_tx * 2 + 1];
            for (int i = 0; i < REG_PER_THREAD; i++) {
                for (int j = 0; j < REG_PER_THREAD; j++) {
                    tmp[i * REG_PER_THREAD + j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < REG_PER_THREAD; i++) {
        for (int j = 0; j < REG_PER_THREAD; j++) {
            C(block_base_y + reg_ty * REG_PER_THREAD + i, block_base_x + reg_tx * REG_PER_THREAD + j) = tmp[i * REG_PER_THREAD + j];
        }
    }
}

// CUTLASS GPU 实现
static void gemm_hand(float *dC, float *dA, float *dB, int M, int N, int K) {

    constexpr int BLOCK_SIZE = 16;
    constexpr int STRIDE = 1;
    constexpr int TILE = BLOCK_SIZE * STRIDE;

    constexpr int M_NUM_PER_BLOCK = 16;
    constexpr int N_NUM_PER_BLOCK = 16;
    constexpr int K_NUM_PER_BLOCK = 16;
    constexpr int NUM_PER_THREAD = 4;

    constexpr int REG_PER_THREAD = 2;

    dim3 block(M_NUM_PER_BLOCK / NUM_PER_THREAD, N_NUM_PER_BLOCK);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    frame_gemm_demo5<M_NUM_PER_BLOCK, N_NUM_PER_BLOCK, K_NUM_PER_BLOCK, NUM_PER_THREAD, REG_PER_THREAD><<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

static void gemm_launch(float *dC, const float *dA, const float *dB, int M, int N, int K) {
    gemm_hand(dC, const_cast<float *>(dA), const_cast<float *>(dB), M, N, K);
}

LC_REGISTER_GEMM_OP("frame_gemm_demo5", opfw::cpu_gemm_f32, gemm_launch);
