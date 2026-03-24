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
template <unsigned int thread_num_x, unsigned int thread_num_y>
__global__ void transpose_demo3(float *A, float *B, int M, int N) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x; // col
    int ty = blockIdx.y * blockDim.y + threadIdx.y; // row

    alignas(16) float src_reg[4][4];
    alignas(16) float dst_reg[4][4];

    for (int i = 0; i < 4; i++) {
        FETCH_FLOAT4(src_reg[i][0]) = FETCH_FLOAT4(A[(ty * thread_num_y + i) * N + (tx * thread_num_x)]);
    }

    for (int i = 0; i < 4; i++) {
        FETCH_FLOAT4(dst_reg[i][0]) = make_float4(src_reg[0][i], src_reg[1][i], src_reg[2][i], src_reg[3][i]);
    }

    for (int i = 0; i < 4; i++) {
        // FETCH_FLOAT4(B[(tx * thread_num_x + i) * M + ty]) = FETCH_FLOAT4(dst_reg[i][0]);     // 这个会崩溃掉，因为访问的内存不是内存对齐的
        FETCH_FLOAT4(B[(tx * thread_num_x + i) * M + ty * thread_num_y]) = FETCH_FLOAT4(dst_reg[i][0]);  // 这个是正常运行的
    }
}

static void transpose_hand(float *dOut, float *dIn, int M, int N) {

    // 通过改变block的形状来达到合并访存的目的
    constexpr unsigned int block_x = 16;
    constexpr unsigned int block_y = 32;
    constexpr unsigned int num_threads_x = 4;
    constexpr unsigned int num_threads_y = 4;

    dim3 block1(block_x, block_y);
    dim3 grid1(N/ (block_x * num_threads_x), M / (block_y * num_threads_y));

    transpose_demo3<num_threads_x, num_threads_y><<<grid1, block1>>>(dIn, dOut, M, N);
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

LC_REGISTER_TRANSPOSE_OP("transpose_demo3", transpose_cpu, transpose_launch);