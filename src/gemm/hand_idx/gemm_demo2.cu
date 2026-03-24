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
static void gemm_cpu(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.f;
            for (int k = 0; k < K; k++) {
                acc += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = acc;
        }
    }
}
// 29.2170887ms
// shared memory:
// read:
// write:
// global memory:
// read:     (M/bm)*(N/bn)个块，每个块(bm*K+K*bn)次访存，总计：KMN(1/bm+1/bn)次
// write:    MN
template <unsigned int BLOCK_SIZE>
__global__ void gemm_demo2(const float *dA, const float *dB, float *dC, int M, int N, int K) {

    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    extern __shared__ float smem[];

    float *A_shared = smem;
    float *B_shared = smem + BLOCK_SIZE * BLOCK_SIZE;

    float tmp = 0.0f;
    for (int s = 0; s < K / BLOCK_SIZE; s++) {

        int A_shared_idx = ty * BLOCK_SIZE + tx;
        int B_shared_idx = ty * BLOCK_SIZE + tx;
        int A_idx = row * K + tx + s * BLOCK_SIZE;
        int B_idx = (ty + s * BLOCK_SIZE) * N + col;

        A_shared[A_shared_idx] = dA[A_idx];
        B_shared[B_shared_idx] = dB[B_idx];
        __syncthreads();
        for (int i = 0; i < BLOCK_SIZE; i++) {
            int A_shared_idx = ty * BLOCK_SIZE + i;
            int B_shared_idx = i * BLOCK_SIZE + tx;
            tmp += A_shared[A_shared_idx] * B_shared[B_shared_idx];
        }
        __syncthreads();
    }
    int idx = row * N + col;
    dC[idx] = tmp;
}

// CUTLASS GPU 实现
static void gemm_hand(float *dC, const float *dA, const float *dB, int M, int N, int K) {

    dim3 block(16, 16);
    // grid 按输出矩阵 C 的 (M,N) 覆盖
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    int shared_size = 2 * 16 * 16 * sizeof(float);
    gemm_demo2<16><<<grid, block, shared_size>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

LC_REGISTER_GEMM_OP("gemm_demo2", gemm_cpu, gemm_hand);
