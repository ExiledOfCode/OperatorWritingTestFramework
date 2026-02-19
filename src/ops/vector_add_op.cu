#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "op_registry.hpp"
#include "utils.cuh"

__global__ void vector_add_kernel(const float *a, const float *b, float *c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

static void vector_add_cuda(const float *d_a, const float *d_b, float *d_c, size_t n) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    vector_add_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
}

static void vector_add_cpu(const float *a, const float *b, float *c, size_t n) {
    for (size_t i = 0; i < n; ++i)
        c[i] = a[i] + b[i];
}

// ======= 你要的：两个“启动函数” =======

static CorrectnessResult run_vector_add_correctness() {
    size_t n = 1 << 20;

    std::vector<float> h_a(n), h_b(n), h_ref(n), h_out(n);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (size_t i = 0; i < n; ++i) {
        h_a[i] = dist(gen);
        h_b[i] = dist(gen);
    }

    vector_add_cpu(h_a.data(), h_b.data(), h_ref.data(), n);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    vector_add_cuda(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost));

    float max_abs = 0.f;
    for (size_t i = 0; i < n; i++) {
        max_abs = std::max(max_abs, std::abs(h_ref[i] - h_out[i]));
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    double thr = 1e-5;
    return CorrectnessResult{max_abs <= thr, (double)max_abs, "max_abs_diff", thr, ""};
}

static PerfResult run_vector_add_perf() {
    size_t n = 1 << 26;

    std::vector<float> h_a(n, 1.0f), h_b(n, 2.0f);
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    // 计算输入数据大小（字节）
    size_t input_size = 2 * n * sizeof(float); // A 和 B 两个向量

    // 分配 GPU 内存
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)));

    // 拷贝数据到 GPU
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // 预热：进行一些内核调用
    for (int i = 0; i < 20; i++)
        vector_add_cuda(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 开始计时
    int iters = 20;
    float ms = cuda_time_ms([&]() {
                   for (int i = 0; i < iters; i++)
                       vector_add_cuda(d_a, d_b, d_c, n);
                   CUDA_CHECK(cudaDeviceSynchronize());
               }) /
               iters;

    // 输出数据大小（字节）
    size_t output_size = n * sizeof(float); // C 向量

    // 计算带宽（GB/s）
    double gbps = 3.0 * n * sizeof(float) / (ms / 1000.0) / 1e9;

    // 计算 CPU 执行时间（假设这里是计算 CPU 上的执行时间，使用与 GPU
    // 相同的内核实现） 这里只是示例，如果需要你可以将 `vector_add_cpu`
    // 的执行时间计入
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        vector_add_cpu(h_a.data(), h_b.data(), h_b.data(), n); // 用 CPU 计算
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = end_time - start_time;
    double cpu_ms = cpu_duration.count() * 1000.0 / iters; // 转换为毫秒

    // 释放 GPU 内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // 返回性能测试结果
    return PerfResult{(double)ms,                         // GPU 执行时间
                      cpu_ms,                             // CPU 执行时间
                      "GB/s",                             // 性能单位
                      gbps,                               // 带宽值
                      "bytes = read(A)+read(B)+write(C)", // 其他说明
                      input_size,                         // 输入数据大小（字节）
                      output_size,                        // 输出数据大小（字节）
                      {{1, n}},
                      {{1, n}}};
}

// ======= 一行注册（你想要的“宏包裹”） =======
REGISTER_OP_FUNCS("vector_add", run_vector_add_correctness, run_vector_add_perf);
