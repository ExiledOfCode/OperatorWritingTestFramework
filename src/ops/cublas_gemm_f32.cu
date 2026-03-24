#include <algorithm>
#include <chrono> // 用于计时
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "op_registry.hpp"
#include "op_runtime.hpp"
#include "utils.cuh"
// cuBLAS GEMM（支持你的 row-major A(MxK), B(KxN), C(MxN)）
// 关键点：cuBLAS按column-major理解内存，所以计算：C_col(NxM) = B_col(NxK) * A_col(KxM)
// 等价于 row-major: C(MxN) = A(MxK) * B(KxN)

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
static void gemm_cublas(float *dC, const float *dA, const float *dB, int M, int N, int K) {
    opfw::gpu_cublas_gemm_f32(dC, dA, dB, M, N, K);
}

LC_REGISTER_GEMM_OP_EX("cublas_gemm_f32", opfw::cpu_gemm_f32, gemm_cublas, [] {
    opfw::GemmSpec spec;
    spec.baselines = {};
    return spec;
}());
