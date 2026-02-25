#include <cublas_v2.h>
#include <algorithm>
#include <chrono> // 用于计时
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "op_registry.hpp"
#include "utils.cuh"
// cuBLAS GEMM（支持你的 row-major A(MxK), B(KxN), C(MxN)）
// 关键点：cuBLAS按column-major理解内存，所以计算：C_col(NxM) = B_col(NxK) * A_col(KxM)
// 等价于 row-major: C(MxN) = A(MxK) * B(KxN)

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
static void gemm_cublas(float *dC, const float *dA, const float *dB, int M, int N, int K) {
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasStatus_t st = cublasCreate(&handle);
        if (st != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cublasCreate failed, status = " << int(st) << "\n";
            std::exit(EXIT_FAILURE);
        }

        // 可选：如果你想更快（A100上常用TF32 TensorCore），但会引入少量数值误差
        // cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
        //
        // 若你想严格FP32路径（更慢但更准），就别设置 math mode（或设 DEFAULT）。
    }

    float alpha = 1.0f;
    float beta  = 0.0f;

    // 这里的 m,n,k 是按 column-major 的矩阵维度：
    // C_col 是 (N x M)，B_col 是 (N x K)，A_col 是 (K x M)
    // 所以：m=N, n=M, k=K
    cublasStatus_t st = cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        /*m=*/N, /*n=*/M, /*k=*/K,
        &alpha,
        /*A=*/dB, /*lda=*/N,   // B_col: (N x K)
        /*B=*/dA, /*ldb=*/K,   // A_col: (K x M)
        &beta,
        /*C=*/dC, /*ldc=*/N    // C_col: (N x M) -> same memory as row-major C(MxN)
    );

    if (st != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasSgemm failed, status = " << int(st) << "\n";
        std::exit(EXIT_FAILURE);
    }
}

static CorrectnessResult run_cublas_gemm_f32_correctness() {
    int M = 256, N = 256, K = 256;

    std::vector<float> hA((size_t)M * K), hB((size_t)K * N), hRef((size_t)M * N), hOut((size_t)M * N);

    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto &x : hA) x = dist(gen);
    for (auto &x : hB) x = dist(gen);

    gemm_cpu(hA.data(), hB.data(), hRef.data(), M, N, K);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, hOut.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

    gemm_cublas(dC, dA, dB, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hOut.data(), dC, hOut.size() * sizeof(float), cudaMemcpyDeviceToHost));

    float max_abs = 0.f;
    for (int i = 0; i < M * N; i++) {
        max_abs = std::max(max_abs, std::abs(hRef[i] - hOut[i]));
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    // 如果你打开了 TF32 math mode，阈值可能需要放宽一些（比如 1e-1 ~ 1e-2 视数据而定）
    double thr = 1e-2;
    return CorrectnessResult{max_abs <= thr, (double)max_abs, "max_abs_diff", thr, ""};
}

static PerfResult run_cublas_gemm_f32_perf() {
    size_t M = 4096, N = 4096, K = 4096;
    std::vector<float> hA((size_t)M * K, 1.f), hB((size_t)K * N, 1.f);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    size_t input_size  = (size_t)(M * K + K * N) * sizeof(float);
    size_t output_size = (size_t)(M * N) * sizeof(float);

    CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, output_size));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

    // 预热
    for (int i = 0; i < 2; i++) gemm_cublas(dC, dA, dB, (int)M, (int)N, (int)K);
    CUDA_CHECK(cudaDeviceSynchronize());

    int iters = 10;
    double ms = cuda_time_ms([&]() {
        for (int i = 0; i < iters; i++) gemm_cublas(dC, dA, dB, (int)M, (int)N, (int)K);
        CUDA_CHECK(cudaDeviceSynchronize());
    }) / iters;

    double flops  = 2.0 * (double)M * (double)N * (double)K;
    double tflops = flops / (ms / 1000.0) / 1e12;

    // CPU时间你这里本来就不测也行，保持一致返回一个测量值
    double cpu_ms = 0.0;

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return PerfResult{
        (double)ms,
        cpu_ms,
        "TFLOP/s",
        tflops,
        "FLOPs = 2*M*N*K",
        input_size,
        output_size,
        {{M, K}, {K, N}},
        {{M, N}}
    };
}

// 注册
REGISTER_OP_FUNCS("cublas_gemm_f32", run_cublas_gemm_f32_correctness, run_cublas_gemm_f32_perf);