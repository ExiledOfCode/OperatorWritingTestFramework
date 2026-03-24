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

template <unsigned int M_NUM_PER_BLOCK, unsigned int N_NUM_PER_BLOCK, unsigned int K_NUM_PER_BLOCK, unsigned int NUM_PER_THREAD>
__global__ void frame_gemm_demo4(float *dA, float *dB, float *dC, int M, int N, int K) {

    MatrixView<float> A{dA, K};
    MatrixView<float> B{dB, N};
    MatrixView<float> C{dC, N};

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int block_base_x = (blockDim.x * NUM_PER_THREAD) * blockIdx.x;
    int block_base_y = blockDim.y * blockIdx.y;

    // int col = tx + block_base_x;
    int row = ty + block_base_y;

    __shared__ float A_shared[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float B_shared[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];

    float tmp[NUM_PER_THREAD] = {0.0f};
    for (int s = 0; s < K / K_NUM_PER_BLOCK; s++) {
        const int k_base = s * K_NUM_PER_BLOCK;
        FETCH_FLOAT4(A_shared[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(A(row, tx * NUM_PER_THREAD + k_base));
        FETCH_FLOAT4(B_shared[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(B(ty + k_base, block_base_x + NUM_PER_THREAD * tx));

        __syncthreads();

        for (int j = 0; j < NUM_PER_THREAD; j++) {
            for (int i = 0; i < K_NUM_PER_BLOCK; i++) {
                tmp[j] += A_shared[ty][i] * B_shared[i][tx * NUM_PER_THREAD + j];
            }
        }
        __syncthreads();
    }
    FETCH_FLOAT4(C(row, block_base_x + tx * NUM_PER_THREAD)) = FETCH_FLOAT4(tmp[0]);
}

// CUTLASS GPU 实现
static void gemm_hand(float *dC, float *dA, float *dB, int M, int N, int K) {

    constexpr int BLOCK_SIZE = 16;
    constexpr int STRIDE = 1;
    constexpr int TILE = BLOCK_SIZE * STRIDE;

    constexpr int N_NUM_PER_BLOCK = 16;
    constexpr int M_NUM_PER_BLOCK = 16;
    constexpr int K_NUM_PER_BLOCK = 16;
    constexpr int NUM_PER_THREAD = 4;

    dim3 block(M_NUM_PER_BLOCK / 4, N_NUM_PER_BLOCK);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    frame_gemm_demo4<M_NUM_PER_BLOCK, N_NUM_PER_BLOCK, K_NUM_PER_BLOCK, NUM_PER_THREAD><<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

static void gemm_launch(float *dC, const float *dA, const float *dB, int M, int N, int K) {
    gemm_hand(dC, const_cast<float *>(dA), const_cast<float *>(dB), M, N, K);
}

LC_REGISTER_GEMM_OP("frame_gemm_demo4", opfw::cpu_gemm_f32, gemm_launch);
