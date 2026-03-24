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
// 60.92079926ms
// global memory: Read: 2MNK, Write: MN
__global__ void gemm_demo1(const float *dA, const float *dB, float *dC, int M, int N, int K) {

    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    int idx = row * N + col;
    if (row < M && col < N) {

        float tmp = 0.0f;
        for (int i = 0; i < K; i++) {
            int dA_idx = row * K + i;
            int dB_idx = i * N + col;
            tmp += dA[dA_idx] * dB[dB_idx];
        }
        dC[idx] = tmp;
    }
}

static void gemm_hand(float *dC, const float *dA, const float *dB, int M, int N, int K) {

    dim3 block(16, 16);
    // grid 按输出矩阵 C 的 (M,N) 覆盖
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemm_demo1<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

LC_REGISTER_GEMM_OP("gemm_demo1", opfw::cpu_gemm_f32, gemm_hand);
