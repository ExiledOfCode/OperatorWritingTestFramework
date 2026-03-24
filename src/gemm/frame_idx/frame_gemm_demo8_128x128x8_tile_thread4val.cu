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
__global__ void frame_gemm_demo8_128x128x8_tile_thread4val(float *dA, float *dB, float *dC, int M, int N, int K) {

    MatrixView<float> A{dA, K};
    MatrixView<float> B{dB, N};
    MatrixView<float> C{dC, N};

    __shared__ float shared_A[2][BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_M];
    __shared__ float shared_B[2][BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_N];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx; // 0..255

    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int block_tile_C_start_x = bx * BLOCK_TILE_SIZE_N;
    const int block_tile_C_start_y = by * BLOCK_TILE_SIZE_M;

    const int thread_tile_C_start_x = block_tile_C_start_x + tx * THREAD_NUM_X;
    const int thread_tile_C_start_y = block_tile_C_start_y + ty * THREAD_NUM_Y;

    const int thread_shared_tile_A_start_y = ty * THREAD_NUM_Y; // 0,8,16,...120
    const int thread_shared_tile_B_start_x = tx * THREAD_NUM_X; // 0,8,16,...120

    alignas(16) float reg_A[THREAD_NUM_Y] = {0.0f};
    alignas(16) float reg_B[THREAD_NUM_X] = {0.0f};
    float tmp[THREAD_NUM_Y][THREAD_NUM_X] = {0.0f};

    int cnt = 0;

    // -----------------------------
    // preload A tile for k_base = 0
    // A tile shape: 128 x 8
    // 128 rows * (8/4=2 float4 per row) = 256 float4
    // 每个线程 1 个 float4
    // -----------------------------
    const int shared_A_row = tid >> 1;         // tid / 2, 0..127
    const int shared_A_col = (tid & 1) << 2;   // (tid % 2) * 4, 0 or 4

    const int shared_B_row = tid >> 5;         // tid / 32, 0..7
    const int shared_B_col = (tid & 31) << 2;  // (tid % 32) * 4, 0..124

    float4 a_vec = FETCH_FLOAT4(A(block_tile_C_start_y + shared_A_row, shared_A_col));

    shared_A[0][shared_A_col + 0][shared_A_row] = a_vec.x;
    shared_A[0][shared_A_col + 1][shared_A_row] = a_vec.y;
    shared_A[0][shared_A_col + 2][shared_A_row] = a_vec.z;
    shared_A[0][shared_A_col + 3][shared_A_row] = a_vec.w;

    // -----------------------------
    // preload B tile for k_base = 0
    // B tile shape: 8 x 128
    // 8 rows * (128/4=32 float4 per row) = 256 float4
    // 每个线程 1 个 float4
    // -----------------------------

    FETCH_FLOAT4(shared_B[0][shared_B_row][shared_B_col]) = FETCH_FLOAT4(B(shared_B_row, block_tile_C_start_x + shared_B_col));

    __syncthreads();

    for (int k_base = BLOCK_TILE_SIZE_K; k_base < K; k_base += BLOCK_TILE_SIZE_K) {

        float4 a_vec = FETCH_FLOAT4(A(block_tile_C_start_y + shared_A_row, k_base + shared_A_col));
        shared_A[cnt ^ 1][shared_A_col + 0][shared_A_row] = a_vec.x;
        shared_A[cnt ^ 1][shared_A_col + 1][shared_A_row] = a_vec.y;
        shared_A[cnt ^ 1][shared_A_col + 2][shared_A_row] = a_vec.z;
        shared_A[cnt ^ 1][shared_A_col + 3][shared_A_row] = a_vec.w;

        FETCH_FLOAT4(shared_B[cnt ^ 1][shared_B_row][shared_B_col]) = FETCH_FLOAT4(B(k_base + shared_B_row, block_tile_C_start_x + shared_B_col));

        // compute current tile
        for (int k = 0; k < BLOCK_TILE_SIZE_K; ++k) {
            for (int i = 0; i < THREAD_NUM_Y; i += 4) {
                FETCH_FLOAT4(reg_A[i]) = FETCH_FLOAT4(shared_A[cnt][k][thread_shared_tile_A_start_y + i]);
            }
            for (int j = 0; j < THREAD_NUM_X; j += 4) {
                FETCH_FLOAT4(reg_B[j]) = FETCH_FLOAT4(shared_B[cnt][k][thread_shared_tile_B_start_x + j]);
            }

            for (int i = 0; i < THREAD_NUM_Y; ++i) {
                for (int j = 0; j < THREAD_NUM_X; ++j) {
                    tmp[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }

        cnt ^= 1;
        __syncthreads();
    }

    // compute last tile
    for (int k = 0; k < BLOCK_TILE_SIZE_K; ++k) {
        for (int i = 0; i < THREAD_NUM_Y; i += 4) {
            FETCH_FLOAT4(reg_A[i]) = FETCH_FLOAT4(shared_A[cnt][k][thread_shared_tile_A_start_y + i]);
        }
        for (int j = 0; j < THREAD_NUM_X; j += 4) {
            FETCH_FLOAT4(reg_B[j]) = FETCH_FLOAT4(shared_B[cnt][k][thread_shared_tile_B_start_x + j]);
        }

        for (int i = 0; i < THREAD_NUM_Y; ++i) {
            for (int j = 0; j < THREAD_NUM_X; ++j) {
                tmp[i][j] += reg_A[i] * reg_B[j];
            }
        }
    }

    // store C
    for (int i = 0; i < THREAD_NUM_Y; ++i) {
        for (int j = 0; j < THREAD_NUM_X; j += 4) {
            FETCH_FLOAT4(C(thread_tile_C_start_y + i, thread_tile_C_start_x + j)) = FETCH_FLOAT4(tmp[i][j]);
        }
    }
}

// launcher
static void gemm_hand(float *dC, float *dA, float *dB, int M, int N, int K) {
    constexpr int BLOCK_TILE_SIZE_M = 128;
    constexpr int BLOCK_TILE_SIZE_N = 128;
    constexpr int BLOCK_TILE_SIZE_K = 8;

    constexpr int THREAD_SIZE_X = 8;
    constexpr int THREAD_SIZE_Y = 8;

    dim3 block(BLOCK_TILE_SIZE_N / THREAD_SIZE_X, BLOCK_TILE_SIZE_M / THREAD_SIZE_Y); // 16 x 16
    dim3 grid((N + BLOCK_TILE_SIZE_N - 1) / BLOCK_TILE_SIZE_N, (M + BLOCK_TILE_SIZE_M - 1) / BLOCK_TILE_SIZE_M);

    frame_gemm_demo8_128x128x8_tile_thread4val<BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_K, THREAD_SIZE_X, THREAD_SIZE_Y>
        <<<grid, block>>>(dA, dB, dC, M, N, K);

    CUDA_CHECK(cudaGetLastError());
}

static void gemm_launch(float *dC, const float *dA, const float *dB, int M, int N, int K) {
    gemm_hand(dC, const_cast<float *>(dA), const_cast<float *>(dB), M, N, K);
}

LC_REGISTER_GEMM_OP("frame_gemm_demo8_128x128x8_tile_thread4val", opfw::cpu_gemm_f32, gemm_launch);
