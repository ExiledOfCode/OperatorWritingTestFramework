#include <algorithm>
#include <chrono> // 用于计时
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "op_registry.hpp"
#include "op_runtime.hpp"
#include "utils.cuh"

template <typename T>
struct MatrixView {
    T *ptr;
    int width;
    __device__ __forceinline__ T &operator()(int r, int c) { return ptr[r * width + c]; }
};

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

static inline int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

static inline int select_swizzle_size(int grid_tiles_n) {
    if (grid_tiles_n >= 6) {
        return 8;
    }
    if (grid_tiles_n >= 3) {
        return 4;
    }
    if (grid_tiles_n >= 2) {
        return 2;
    }
    return 1;
}

template <unsigned int BLOCK_TILE_SIZE_M, unsigned int BLOCK_TILE_SIZE_N, unsigned int BLOCK_TILE_SIZE_K, unsigned int THREAD_NUM_X, unsigned int THREAD_NUM_Y>
__global__ void frame_gemm_demo8_128x128x8_tile_thread4val_swizzle(
    float *dA,
    float *dB,
    float *dC,
    int M,
    int N,
    int K,
    int grid_tiles_m,
    int grid_tiles_n,
    int swizzle_size) {

    MatrixView<float> A{dA, K};
    MatrixView<float> B{dB, N};
    MatrixView<float> C{dC, N};

    __shared__ float shared_A[2][BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_M];
    __shared__ float shared_B[2][BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_N];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx; // 0..255

    // CTA swizzle:
    // 沿着 grid.x 先遍历 M tile，同时把 N tile 按小组折叠到 grid.y，提升相邻 CTA 对 B tile 的复用概率。
    const int block_tile_m = blockIdx.x / swizzle_size;
    const int block_tile_n = blockIdx.y * swizzle_size + (blockIdx.x % swizzle_size);

    if (block_tile_m >= grid_tiles_m || block_tile_n >= grid_tiles_n) {
        return;
    }

    const int block_tile_C_start_x = block_tile_n * BLOCK_TILE_SIZE_N;
    const int block_tile_C_start_y = block_tile_m * BLOCK_TILE_SIZE_M;

    const int thread_tile_C_start_x = block_tile_C_start_x + tx * THREAD_NUM_X;
    const int thread_tile_C_start_y = block_tile_C_start_y + ty * THREAD_NUM_Y;

    const int thread_shared_tile_A_start_y = ty * THREAD_NUM_Y; // 0,8,16,...120
    const int thread_shared_tile_B_start_x = tx * THREAD_NUM_X; // 0,8,16,...120

    alignas(16) float reg_A[THREAD_NUM_Y] = {0.0f};
    alignas(16) float reg_B[THREAD_NUM_X] = {0.0f};
    float tmp[THREAD_NUM_Y][THREAD_NUM_X] = {0.0f};

    int cnt = 0;

    // A tile: 128 x 8 -> 256 float4, each thread loads one float4.
    const int shared_A_row = tid >> 1;       // tid / 2, 0..127
    const int shared_A_col = (tid & 1) << 2; // (tid % 2) * 4, 0 or 4

    // B tile: 8 x 128 -> 256 float4, each thread loads one float4.
    const int shared_B_row = tid >> 5;       // tid / 32, 0..7
    const int shared_B_col = (tid & 31) << 2; // (tid % 32) * 4, 0..124

    float4 a_vec = FETCH_FLOAT4(A(block_tile_C_start_y + shared_A_row, shared_A_col));

    shared_A[0][shared_A_col + 0][shared_A_row] = a_vec.x;
    shared_A[0][shared_A_col + 1][shared_A_row] = a_vec.y;
    shared_A[0][shared_A_col + 2][shared_A_row] = a_vec.z;
    shared_A[0][shared_A_col + 3][shared_A_row] = a_vec.w;

    FETCH_FLOAT4(shared_B[0][shared_B_row][shared_B_col]) =
        FETCH_FLOAT4(B(shared_B_row, block_tile_C_start_x + shared_B_col));

    __syncthreads();

    for (int k_base = BLOCK_TILE_SIZE_K; k_base < K; k_base += BLOCK_TILE_SIZE_K) {
        float4 next_a_vec = FETCH_FLOAT4(A(block_tile_C_start_y + shared_A_row, k_base + shared_A_col));
        shared_A[cnt ^ 1][shared_A_col + 0][shared_A_row] = next_a_vec.x;
        shared_A[cnt ^ 1][shared_A_col + 1][shared_A_row] = next_a_vec.y;
        shared_A[cnt ^ 1][shared_A_col + 2][shared_A_row] = next_a_vec.z;
        shared_A[cnt ^ 1][shared_A_col + 3][shared_A_row] = next_a_vec.w;

        FETCH_FLOAT4(shared_B[cnt ^ 1][shared_B_row][shared_B_col]) =
            FETCH_FLOAT4(B(k_base + shared_B_row, block_tile_C_start_x + shared_B_col));

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

    for (int i = 0; i < THREAD_NUM_Y; ++i) {
        for (int j = 0; j < THREAD_NUM_X; j += 4) {
            FETCH_FLOAT4(C(thread_tile_C_start_y + i, thread_tile_C_start_x + j)) = FETCH_FLOAT4(tmp[i][j]);
        }
    }
}

static void gemm_hand(float *dC, float *dA, float *dB, int M, int N, int K) {
    constexpr int BLOCK_TILE_SIZE_M = 128;
    constexpr int BLOCK_TILE_SIZE_N = 128;
    constexpr int BLOCK_TILE_SIZE_K = 8;

    constexpr int THREAD_SIZE_X = 8;
    constexpr int THREAD_SIZE_Y = 8;

    const int grid_tiles_m = ceil_div(M, BLOCK_TILE_SIZE_M);
    const int grid_tiles_n = ceil_div(N, BLOCK_TILE_SIZE_N);
    const int swizzle_size = select_swizzle_size(grid_tiles_n);

    dim3 block(BLOCK_TILE_SIZE_N / THREAD_SIZE_X, BLOCK_TILE_SIZE_M / THREAD_SIZE_Y); // 16 x 16
    dim3 grid(grid_tiles_m * swizzle_size, ceil_div(grid_tiles_n, swizzle_size));

    frame_gemm_demo8_128x128x8_tile_thread4val_swizzle<
        BLOCK_TILE_SIZE_M,
        BLOCK_TILE_SIZE_N,
        BLOCK_TILE_SIZE_K,
        THREAD_SIZE_X,
        THREAD_SIZE_Y><<<grid, block>>>(dA, dB, dC, M, N, K, grid_tiles_m, grid_tiles_n, swizzle_size);

    CUDA_CHECK(cudaGetLastError());
}

static void gemm_launch(float *dC, const float *dA, const float *dB, int M, int N, int K) {
    gemm_hand(dC, const_cast<float *>(dA), const_cast<float *>(dB), M, N, K);
}

LC_REGISTER_GEMM_OP("frame_gemm_demo8_128x128x8_tile_thread4val_swizzle", opfw::cpu_gemm_f32, gemm_launch);
