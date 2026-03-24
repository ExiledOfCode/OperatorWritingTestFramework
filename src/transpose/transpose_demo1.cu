#include <algorithm>
#include <chrono> // 用于计时
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "op_registry.hpp"
#include "op_runtime.hpp"
#include "utils.cuh"

// ============================
// Naive Transpose Kernel
// ============================
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
__global__ void transpose_demo1_kernel(const float *A, float *B, int M, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // col
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row

    if (x < N && y < M) {
        B[x * M + y] = A[y * N + x];
    }
}
static void transpose_hand(float *dOut, const float *dIn, int M, int N) {

    // 通过改变block的形状来达到合并访存的目的
    dim3 block1(32, 8);
    dim3 block2(16, 16);
    dim3 block3(8, 32);
    dim3 grid1((N + 31) / 32, (M + 7) / 8);
    dim3 grid2((N + 15) / 16, (M + 15) / 16);
    dim3 grid3((N + 7) / 8, (M + 31) / 32);

    transpose_demo1_kernel<<<grid1, block1>>>(dIn, dOut, M, N);
    transpose_demo1_kernel<<<grid2, block2>>>(dIn, dOut, M, N);
    transpose_demo1_kernel<<<grid3, block3>>>(dIn, dOut, M, N);
    CUDA_CHECK(cudaGetLastError());
}

static void transpose_cpu(const float *A, float *B, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            B[j * M + i] = A[i * N + j];
        }
    }
}

LC_REGISTER_TRANSPOSE_OP("transpose_demo1", transpose_cpu, transpose_hand);