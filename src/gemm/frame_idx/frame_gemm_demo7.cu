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
__global__ void frame_gemm_demo7(float *dA, float *dB, float *dC, int M, int N, int K) {

    MatrixView<float> A{dA, K};
    MatrixView<float> B{dB, N};
    MatrixView<float> C{dC, N};

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int block_base_x = (blockDim.x * NUM_PER_THREAD) * blockIdx.x;
    int block_base_y = (blockDim.y * NUM_PER_THREAD) * blockIdx.y;

    __shared__ float A_shared[K_NUM_PER_BLOCK][M_NUM_PER_BLOCK];
    __shared__ float B_shared[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];

    float a_reg[REG_PER_THREAD] = {0.0f};
    float b_reg[REG_PER_THREAD] = {0.0f};

    // 复用a_reg来将数据从全局读到寄存器里做过渡
    float *a_load_shared_reg = a_reg;

    float tmp[REG_PER_THREAD][REG_PER_THREAD] = {0.0f};
    int shared_compute_base_y = ty * NUM_PER_THREAD;
    int shared_compute_base_x = tx * NUM_PER_THREAD;

    int blcok_compute_base_y = block_base_y + shared_compute_base_y;
    int block_compute_base_x = block_base_x + shared_compute_base_x;

    for (int s = 0; s < K / K_NUM_PER_BLOCK; s++) {
        const int k_base = s * K_NUM_PER_BLOCK;
        // 从全局内存向shared内存中搬运元素
        for (int i = 0; i < NUM_PER_THREAD; i++) {
            FETCH_FLOAT4(a_load_shared_reg[0]) = FETCH_FLOAT4(A(blcok_compute_base_y + i, shared_compute_base_x + k_base));

            for (int j = 0; j < REG_PER_THREAD; j++) {
                A_shared[shared_compute_base_x + j][shared_compute_base_y + i] = a_load_shared_reg[j];
            }
            FETCH_FLOAT4(B_shared[shared_compute_base_y + i][shared_compute_base_x]) =
                FETCH_FLOAT4(B(shared_compute_base_y + k_base + i, block_compute_base_x));
        }
        __syncthreads();
        for (int k = 0; k < K_NUM_PER_BLOCK; k++) {
            // 从A矩阵拿元素
            FETCH_FLOAT4(a_reg[0]) = FETCH_FLOAT4(A_shared[k][shared_compute_base_y]);
            // 从B矩阵拿元素
            FETCH_FLOAT4(b_reg[0]) = FETCH_FLOAT4(B_shared[k][shared_compute_base_x]);
            // 使用寄存器的元素来进行计算
            for (int i = 0; i < REG_PER_THREAD; i++) {
                for (int j = 0; j < REG_PER_THREAD; j++) {
                    tmp[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }

    // 将寄存器中的元素放回到全局内存中
    for (int i = 0; i < REG_PER_THREAD; i++) {
        for (int j = 0; j < REG_PER_THREAD; j++) {
            C(blcok_compute_base_y + i, block_compute_base_x + j) = tmp[i][j];
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

    constexpr int REG_PER_THREAD = 4;

    dim3 block(M_NUM_PER_BLOCK / NUM_PER_THREAD, N_NUM_PER_BLOCK / NUM_PER_THREAD);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    frame_gemm_demo7<M_NUM_PER_BLOCK, N_NUM_PER_BLOCK, K_NUM_PER_BLOCK, NUM_PER_THREAD, REG_PER_THREAD><<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

static void gemm_launch(float *dC, const float *dA, const float *dB, int M, int N, int K) {
    gemm_hand(dC, const_cast<float *>(dA), const_cast<float *>(dB), M, N, K);
}

LC_REGISTER_GEMM_OP("frame_gemm_demo7", opfw::cpu_gemm_f32, gemm_launch);
