#include <algorithm>
#include <chrono> // 用于计时
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "op_registry.hpp"
#include "op_runtime.hpp"
#include "utils.cuh"

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
__global__ void transpose_demo2_kernel(float *A, float *B, int M, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // col
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row

    float reg[4];
    FETCH_FLOAT4(reg[0]) = FETCH_FLOAT4(A[y * N + x * 4]);

    for (int i = 0; i < 4; i++)
        B[(x * 4 + i) * M + y] = reg[i];
}
static void transpose_hand(float *dOut, float *dIn, int M, int N) {

    // 通过改变block的形状来达到合并访存的目的
    constexpr unsigned int block_x = 2;
    constexpr unsigned int block_y = 128;
    constexpr unsigned int num_threads = 4;

    dim3 block1(block_x, block_y);
    dim3 grid1((N + (block_x * num_threads) - 1) / (block_x * num_threads), (M + block_y - 1) / block_y);

    transpose_demo2_kernel<<<grid1, block1>>>(dIn, dOut, M, N);
    CUDA_CHECK(cudaGetLastError());
}

static void transpose_cpu(const float *A, float *B, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            B[j * M + i] = A[i * N + j];
        }
    }
}

static void transpose_launch(float *dOut, const float *dIn, int M, int N) {
    transpose_hand(dOut, const_cast<float *>(dIn), M, N);
}

LC_REGISTER_TRANSPOSE_OP("transpose_demo2", transpose_cpu, transpose_launch);