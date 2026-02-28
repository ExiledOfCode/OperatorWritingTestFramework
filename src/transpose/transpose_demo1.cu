#include <algorithm>
#include <chrono> // 用于计时
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "op_registry.hpp"
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

// ======= 你要的：两个“启动函数” =======
static CorrectnessResult correctness() {
    int M = 2048;
    int N = 512;

    std::vector<float> hIn((size_t)M * N);
    std::vector<float> hRef((size_t)M * N);
    std::vector<float> hOut((size_t)M * N);

    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    for (auto &x : hIn)
        x = dist(gen) * 10;

    transpose_cpu(hIn.data(), hRef.data(), M, N);

    float *dIn = nullptr;
    float *dOut = nullptr;

    CUDA_CHECK(cudaMalloc(&dIn, hIn.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOut, hOut.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), hIn.size() * sizeof(float), cudaMemcpyHostToDevice));

    transpose_hand(dOut, dIn, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, hOut.size() * sizeof(float), cudaMemcpyDeviceToHost));

    float max_abs = 0.f;
    for (int i = 0; i < M * N; i++) {
        max_abs = std::max(max_abs, std::abs(hRef[i] - hOut[i]));
    }

    // dump_to_csv_any("gemm_dump.csv", {{"dIn", hIn.data(), {M, N}}, {"C_gpu", hOut.data(), {N, M}}, {"C_cpu", hRef.data(), {N, M}}});
    cudaFree(dIn);
    cudaFree(dOut);

    double thr = 1e-5;
    return CorrectnessResult{max_abs <= thr, (double)max_abs, "max_abs_diff", thr, ""};
}

static PerfResult perf() {
    size_t M = 2048;
    size_t N = 512;

    std::vector<float> hIn((size_t)M * N, 1.f);

    float *dIn = nullptr;
    float *dOut = nullptr;

    size_t input_size = M * N * sizeof(float);
    size_t output_size = M * N * sizeof(float);

    CUDA_CHECK(cudaMalloc(&dIn, input_size));
    CUDA_CHECK(cudaMalloc(&dOut, output_size));

    CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), input_size, cudaMemcpyHostToDevice));

    // 预热
    for (int i = 0; i < 0; i++)
        transpose_hand(dOut, dIn, M, N);

    CUDA_CHECK(cudaDeviceSynchronize());

    int iters = 1;

    double ms = cuda_time_ms([&]() {
                    for (int i = 0; i < iters; i++)
                        transpose_hand(dOut, dIn, M, N);
                    CUDA_CHECK(cudaDeviceSynchronize());
                }) /
                iters;

    // transpose FLOPs 很少，这里算 memory bandwidth
    double bytes = 2.0 * input_size; // 读 + 写
    double gbps = bytes / (ms / 1000.0) / 1e9;

    cudaFree(dIn);
    cudaFree(dOut);

    return PerfResult{(double)ms, 0.0, "GB/s", gbps, "Bytes = 2*M*N*sizeof(float)", input_size, output_size, {{M, N}}, {{N, M}}};
}

// ======= 一行注册（你想要的“宏包裹”） =======
REGISTER_OP_FUNCS("transpose_demo1", correctness, perf);