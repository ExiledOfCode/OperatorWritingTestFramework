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

template <unsigned int BLOCK_TILE_SIZE_M, unsigned int BLOCK_TILE_SIZE_N, unsigned int BLOCK_TILE_SIZE_K, unsigned int THREAD_NUM_X, unsigned int THREAD_NUM_Y>
__global__ void A100_gemm_demo8_128x128x8_tile(float *dA, float *dB, float *dC, int M, int N, int K) {

    MatrixView<float> A{dA, K};
    MatrixView<float> B{dB, N};
    MatrixView<float> C{dC, N};

    __shared__ float shared_A[2][BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_M];
    __shared__ float shared_B[2][BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_N];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int block_tile_C_start_x = bx * (blockDim.x * THREAD_NUM_X);
    int block_tile_C_start_y = by * (blockDim.y * THREAD_NUM_Y);

    int thread_tile_C_start_x = block_tile_C_start_x + tx * THREAD_NUM_X;
    int thread_tile_C_start_y = block_tile_C_start_y + ty * THREAD_NUM_Y;

    int thread_shared_tile_A_start_y = ty * THREAD_NUM_Y;
    int thread_shared_tile_B_start_x = tx * THREAD_NUM_X;

    alignas(16) float reg_A[THREAD_NUM_Y] = {0.0f};
    alignas(16) float reg_B[THREAD_NUM_X] = {0.0f};

    float tmp[THREAD_NUM_Y][THREAD_NUM_X] = {0.0f};
    float *reg_load_shared_A = reg_A;

    int cnt = 0;
    for (int j = 0; j < THREAD_NUM_Y; j++) {
        for (int i = 0; i < THREAD_NUM_X; i += 4) {
            FETCH_FLOAT4(reg_load_shared_A[i]) = FETCH_FLOAT4(A(thread_tile_C_start_y + j, i));
        }
        for (int i = 0; i < THREAD_NUM_X; i++) {
            shared_A[0][i][thread_shared_tile_A_start_y + j] = reg_load_shared_A[i];
        }
    }

    for (int j = 0; j < THREAD_NUM_Y; j++) {
        for (int i = 0; i < THREAD_NUM_X; i += 4) {
            FETCH_FLOAT4(shared_B[0][j][thread_shared_tile_B_start_x + i]) = FETCH_FLOAT4(B(j, thread_tile_C_start_x + i));
        }
    }

    __syncthreads();

    for (int k_base = BLOCK_TILE_SIZE_K; k_base < K; k_base += BLOCK_TILE_SIZE_K) {
        for (int j = 0; j < THREAD_NUM_Y; j++) {
            for (int i = 0; i < THREAD_NUM_X; i += 4) {
                FETCH_FLOAT4(reg_load_shared_A[i]) = FETCH_FLOAT4(A(thread_tile_C_start_y + j, k_base + i));
            }
            for (int i = 0; i < THREAD_NUM_X; i++) {
                shared_A[cnt ^ 1][i][thread_shared_tile_A_start_y + j] = reg_load_shared_A[i];
            }
        }

        for (int j = 0; j < THREAD_NUM_Y; j++) {
            for (int i = 0; i < THREAD_NUM_X; i += 4) {
                FETCH_FLOAT4(shared_B[cnt ^ 1][j][thread_shared_tile_B_start_x + i]) = FETCH_FLOAT4(B(k_base + j, thread_tile_C_start_x + i));
            }
        }

        for (int k = 0; k < BLOCK_TILE_SIZE_K; k++) {
            for (int i = 0; i < THREAD_NUM_Y; i += 4) {
                FETCH_FLOAT4(reg_A[i]) = FETCH_FLOAT4(shared_A[cnt][k][thread_shared_tile_A_start_y + i]);
            }
            for (int i = 0; i < THREAD_NUM_X; i += 4) {
                FETCH_FLOAT4(reg_B[i]) = FETCH_FLOAT4(shared_B[cnt][k][thread_shared_tile_B_start_x + i]);
            }
            for (int i = 0; i < THREAD_NUM_Y; i++) {
                for (int j = 0; j < THREAD_NUM_X; j++) {
                    tmp[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }
        cnt ^= 1;
        __syncthreads();
    }
    for (int k = 0; k < BLOCK_TILE_SIZE_K; k++) {
        for (int i = 0; i < THREAD_NUM_Y; i += 4) {
            FETCH_FLOAT4(reg_A[i]) = FETCH_FLOAT4(shared_A[cnt][k][thread_shared_tile_A_start_y + i]);
        }
        for (int i = 0; i < THREAD_NUM_X; i += 4) {
            FETCH_FLOAT4(reg_B[i]) = FETCH_FLOAT4(shared_B[cnt][k][thread_shared_tile_B_start_x + i]);
        }
        for (int i = 0; i < THREAD_NUM_Y; i++) {
            for (int j = 0; j < THREAD_NUM_X; j++) {
                tmp[i][j] += reg_A[i] * reg_B[j];
            }
        }
    }
    __syncthreads();

    for (int i = 0; i < THREAD_NUM_Y; i++) {
        for (int j = 0; j < THREAD_NUM_X; j += 4) {
            FETCH_FLOAT4(C(thread_tile_C_start_y + i, thread_tile_C_start_x + j)) = FETCH_FLOAT4(tmp[i][j]);
        }
    }
}

// CUTLASS GPU 实现
static void gemm_hand(float *dC, float *dA, float *dB, int M, int N, int K) {
    constexpr int BLOCK_TILE_SIZE_M = 128;
    constexpr int BLOCK_TILE_SIZE_N = 128;
    constexpr int BLOCK_TILE_SIZE_K = 8;

    constexpr int THREAD_SIZE_X = 8;
    constexpr int THREAD_SIZE_Y = 8;

    dim3 block(BLOCK_TILE_SIZE_N / THREAD_SIZE_X, BLOCK_TILE_SIZE_M / THREAD_SIZE_Y);
    dim3 grid((N + BLOCK_TILE_SIZE_N - 1) / BLOCK_TILE_SIZE_N, (M + BLOCK_TILE_SIZE_M - 1) / BLOCK_TILE_SIZE_M);

    A100_gemm_demo8_128x128x8_tile<BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_K, THREAD_SIZE_X, THREAD_SIZE_Y><<<grid, block>>>(dA, dB, dC, M, N, K);

    CUDA_CHECK(cudaGetLastError());
}

static void gemm_launch(float *dC, const float *dA, const float *dB, int M, int N, int K) {
    gemm_hand(dC, const_cast<float *>(dA), const_cast<float *>(dB), M, N, K);
}

LC_REGISTER_GEMM_OP("A100_gemm_demo8_128x128x8_tile", opfw::cpu_gemm_f32, gemm_launch);
