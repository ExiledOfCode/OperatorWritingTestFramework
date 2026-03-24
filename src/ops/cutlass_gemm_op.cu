#include <algorithm>
#include <chrono> // 用于计时
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "op_registry.hpp"
#include "op_runtime.hpp"
#include "utils.cuh"

// CUTLASS
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"

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

// CUTLASS GPU 实现
static void gemm_cutlass(float *dC, const float *dA, const float *dB, int M, int N, int K) {
    using Element = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using Gemm = cutlass::gemm::device::Gemm<Element, LayoutA, Element, LayoutB, Element, LayoutC, Element>;

    Element alpha = 1.0f;
    Element beta = 0.0f;

    Gemm gemm_op;
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    typename Gemm::Arguments args(problem_size, {dA, K}, {dB, N}, {dC, N}, {dC, N}, {alpha, beta});

    cutlass::Status status = gemm_op(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM failed, status = " << int(status) << "\n";
        std::exit(EXIT_FAILURE);
    }
}

LC_REGISTER_GEMM_OP_EX("cutlass_gemm_f32", gemm_cpu, gemm_cutlass, [] {
    opfw::GemmSpec spec;
    spec.baselines = {};
    return spec;
}());
