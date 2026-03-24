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
// 展开 14.96444798ms，不展开18.40156746ms
// shared memory:
// read:
// write:   MN
// global memory:
// read:     (M/bm)*(N/bn)个块，每个块(bm*K+K*bn)次访存，总计：KMN(1/bm+1/bn)次
// write:    MN
template <unsigned int BLOCK_SIZE, unsigned int STRIDE>
__global__ void gemm_demo3(const float *dA, const float *dB, float *dC, int M, int N, int K) {

    int block_col = (blockDim.x * STRIDE) * blockIdx.x;
    int block_row = (blockDim.y * STRIDE) * blockIdx.y;
    int col = threadIdx.x + block_col;
    int row = threadIdx.y + block_row;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    extern __shared__ float smem[];
    int TILE = BLOCK_SIZE * STRIDE;

    float *A_shared = smem;
    float *B_shared = smem + TILE * TILE;

    if (block_row + TILE <= M && block_col + TILE <= N) {
        float tmp[STRIDE][STRIDE] = {0.0f};
        for (int s = 0; s < K / TILE; s++) {
            // 全局内存到shared mem
#pragma unroll // 展开 14.96444798ms，不展开18.40156746ms
            for (int y_stride = 0; y_stride < STRIDE; y_stride++) {
                for (int x_stride = 0; x_stride < STRIDE; x_stride++) {
                    int A_shared_idx = (ty + y_stride * BLOCK_SIZE) * BLOCK_SIZE * STRIDE + tx + x_stride * BLOCK_SIZE;
                    int B_shared_idx = (ty + y_stride * BLOCK_SIZE) * BLOCK_SIZE * STRIDE + tx + x_stride * BLOCK_SIZE;
                    int A_idx = (row + BLOCK_SIZE * y_stride) * K + tx + s * TILE + x_stride * BLOCK_SIZE;
                    int B_idx = (ty + s * TILE + BLOCK_SIZE * y_stride) * N + col + x_stride * BLOCK_SIZE;

                    A_shared[A_shared_idx] = dA[A_idx];
                    B_shared[B_shared_idx] = dB[B_idx];
                }
            }
            __syncthreads();
            // shared mem 到寄存器
#pragma unroll
            for (int y_stride = 0; y_stride < STRIDE; y_stride++) {
                for (int x_stride = 0; x_stride < STRIDE; x_stride++) {
                    for (int i = 0; i < TILE; i++) {
                        int A_shared_idx = (ty + y_stride * BLOCK_SIZE) * TILE + i;
                        int B_shared_idx = i * TILE + tx + x_stride * BLOCK_SIZE;
                        tmp[y_stride][x_stride] += A_shared[A_shared_idx] * B_shared[B_shared_idx];
                    }
                }
            }
            __syncthreads();
        }

        // 寄存器写回到内存
#pragma unroll
        for (int y_stride = 0; y_stride < STRIDE; y_stride++) {
            for (int x_stride = 0; x_stride < STRIDE; x_stride++) {
                int idx = (row + y_stride * BLOCK_SIZE) * N + col + x_stride * BLOCK_SIZE;
                dC[idx] = tmp[y_stride][x_stride];
            }
        }
    }
}

// CUTLASS GPU 实现
static void gemm_hand(float *dC, const float *dA, const float *dB, int M, int N, int K) {

    constexpr int BLOCK_SIZE = 16;
    constexpr int STRIDE = 2;
    constexpr int TILE = BLOCK_SIZE * STRIDE;

    int shared_size = 2 * STRIDE * STRIDE * 16 * 16 * sizeof(float);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    gemm_demo3<16, STRIDE><<<grid, block, shared_size>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

LC_REGISTER_GEMM_OP("gemm_demo3", opfw::cpu_gemm_f32, gemm_hand);
