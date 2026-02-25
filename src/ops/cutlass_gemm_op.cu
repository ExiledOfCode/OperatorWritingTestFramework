#include <algorithm>
#include <chrono> // 用于计时
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "op_registry.hpp"
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

// ======= 你要的：两个“启动函数” =======

static CorrectnessResult run_cutlass_gemm_f32_correctness() {
    int M = 256, N = 256, K = 256;

    std::vector<float> hA((size_t)M * K), hB((size_t)K * N), hRef((size_t)M * N), hOut((size_t)M * N);

    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto &x : hA)
        x = dist(gen);
    for (auto &x : hB)
        x = dist(gen);

    gemm_cpu(hA.data(), hB.data(), hRef.data(), M, N, K);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, hOut.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

    gemm_cutlass(dC, dA, dB, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hOut.data(), dC, hOut.size() * sizeof(float), cudaMemcpyDeviceToHost));

    float max_abs = 0.f;
    for (int i = 0; i < M * N; i++) {
        max_abs = std::max(max_abs, std::abs(hRef[i] - hOut[i]));
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    double thr = 1e-2;
    return CorrectnessResult{max_abs <= thr, (double)max_abs, "max_abs_diff", thr, ""};
}

static PerfResult run_cutlass_gemm_f32_perf() {
    size_t M = 4096, N = 4096, K = 4096;
    std::vector<float> hA((size_t)M * K, 1.f), hB((size_t)K * N, 1.f);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    size_t input_size = 2 * M * K * sizeof(float); // 输入数据大小 A 和 B
    size_t output_size = M * N * sizeof(float);    // 输出数据大小 C

    // 分配 GPU 内存
    CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, output_size));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

    // 预热：进行一些内核调用
    for (int i = 0; i < 2; i++)
        gemm_cutlass(dC, dA, dB, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 开始计时：使用 cuda_time_ms 测量 GPU 执行时间
    int iters = 2;
    double ms = cuda_time_ms([&]() {
                    for (int i = 0; i < iters; i++)
                        gemm_cutlass(dC, dA, dB, M, N, K);
                    CUDA_CHECK(cudaDeviceSynchronize());
                }) /
                iters; // cuda_time_ms 返回的是毫秒，所以这里直接除以迭代次数

    // 计算带宽（GB/s）
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double tflops = flops / (ms / 1000.0) / 1e12;

    // 计算 CPU 执行时间
    auto cpu_start_time = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < iters / 2; i++) {
    //     gemm_cpu(hA.data(), hB.data(), hA.data(), M, N, K); // 使用 CPU 计算
    // }
    auto cpu_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = cpu_end_time - cpu_start_time;
    double cpu_ms = cpu_duration.count() * 1000.0 / iters / 2;
    // 释放 GPU 内存
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    // 返回性能测试结果
    return PerfResult{
        (double)ms,        // GPU 执行时间
        cpu_ms,            // CPU 执行时间
        "TFLOP/s",         // 性能单位
        tflops,            // 性能值（计算吞吐量）
        "FLOPs = 2*M*N*K", // 说明信息
        input_size,        // 输入数据大小（字节）
        output_size,       // 输出数据大小（字节）
        {{M, K}, {K, N}},  // 输入数据的形状
        {{M, N}}           // 输出数据的形状
    };
}

// ======= 一行注册（你想要的“宏包裹”） =======
REGISTER_OP_FUNCS("cutlass_gemm_f32", run_cutlass_gemm_f32_correctness, run_cutlass_gemm_f32_perf);
