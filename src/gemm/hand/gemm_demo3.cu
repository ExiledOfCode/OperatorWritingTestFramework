#include <algorithm>
#include <chrono> // 用于计时
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "op_registry.hpp"
#include "utils.cuh"

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
// 展开 14.96444798ms，不展开18.40156746ms
// shared memory:
// read:
// write:   MN
// global memory:
// read:     (M/bm)*(N/bn)个块，每个块(bm*K+K*bn)次访存，总计：KMN(1/bm+1/bn)次
// write:    MN
template <unsigned int BLOCK_SIZE, unsigned int STRIDE>
__global__ void gemm_demo3(const float *dA, const float *dB, float *dC, int M, int N, int K) {

    int block_col = (blockDim.x * STRIDE) * blockIdx.x;
    int block_row = (blockDim.y * STRIDE) * blockIdx.y;
    int col = threadIdx.x + block_col;
    int row = threadIdx.y + block_row;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    extern __shared__ float smem[];
    int TILE = BLOCK_SIZE * STRIDE;

    float *A_shared = smem;
    float *B_shared = smem + TILE * TILE;

    if (block_row + TILE <= M && block_col + TILE <= N) {
        float tmp[STRIDE][STRIDE] = {0.0f};
        for (int s = 0; s < K / TILE; s++) {
            // 全局内存到shared mem
#pragma unroll  // 展开 14.96444798ms，不展开18.40156746ms
            for (int y_stride = 0; y_stride < STRIDE; y_stride++) {
                for (int x_stride = 0; x_stride < STRIDE; x_stride++) {
                    int A_shared_idx = (ty + y_stride * BLOCK_SIZE) * BLOCK_SIZE * STRIDE + tx + x_stride * BLOCK_SIZE;
                    int B_shared_idx = (ty + y_stride * BLOCK_SIZE) * BLOCK_SIZE * STRIDE + tx + x_stride * BLOCK_SIZE;
                    int A_idx = (row + BLOCK_SIZE * y_stride) * K + tx + s * TILE + x_stride * BLOCK_SIZE;
                    int B_idx = (ty + s * TILE + BLOCK_SIZE * y_stride) * N + col + x_stride * BLOCK_SIZE;

                    A_shared[A_shared_idx] = dA[A_idx];
                    B_shared[B_shared_idx] = dB[B_idx];
                }
            }
            __syncthreads();
            // shared mem 到寄存器
#pragma unroll
            for (int y_stride = 0; y_stride < STRIDE; y_stride++) {
                for (int x_stride = 0; x_stride < STRIDE; x_stride++) {
                    for (int i = 0; i < TILE; i++) {
                        int A_shared_idx = (ty + y_stride * BLOCK_SIZE) * TILE + i;
                        int B_shared_idx = i * TILE + tx + x_stride * BLOCK_SIZE;
                        tmp[y_stride][x_stride] += A_shared[A_shared_idx] * B_shared[B_shared_idx];
                    }
                }
            }
            __syncthreads();
        }

        // 寄存器写回到内存
#pragma unroll
        for (int y_stride = 0; y_stride < STRIDE; y_stride++) {
            for (int x_stride = 0; x_stride < STRIDE; x_stride++) {
                int idx = (row + y_stride * BLOCK_SIZE) * N + col + x_stride * BLOCK_SIZE;
                dC[idx] = tmp[y_stride][x_stride];
            }
        }
    }
}

// CUTLASS GPU 实现
static void gemm_hand(float *dC, const float *dA, const float *dB, int M, int N, int K) {

    constexpr int BLOCK_SIZE = 16;
    constexpr int STRIDE = 2;
    constexpr int TILE = BLOCK_SIZE * STRIDE;

    int shared_size = 2 * STRIDE * STRIDE * 16 * 16 * sizeof(float);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    gemm_demo3<16, STRIDE><<<grid, block, shared_size>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

// ======= 你要的：两个“启动函数” =======

static CorrectnessResult correctness() {
    int M = 256, N = 512, K = 512;

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

    gemm_hand(dC, dA, dB, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hOut.data(), dC, hOut.size() * sizeof(float), cudaMemcpyDeviceToHost));

    float max_abs = 0.f;
    for (int i = 0; i < M * N; i++) {
        max_abs = std::max(max_abs, std::abs(hRef[i] - hOut[i]));
    }

    // dump_to_csv("gemm_dump.csv", hA.data(), hB.data(), hOut.data(), hRef.data(), M, N, K);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    double thr = 1e-2;
    return CorrectnessResult{max_abs <= thr, (double)max_abs, "max_abs_diff", thr, ""};
}

static PerfResult perf() {
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
        gemm_hand(dC, dA, dB, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 开始计时：使用 cuda_time_ms 测量 GPU 执行时间
    int iters = 2;
    double ms = cuda_time_ms([&]() {
                    for (int i = 0; i < iters; i++)
                        gemm_hand(dC, dA, dB, M, N, K);
                    CUDA_CHECK(cudaDeviceSynchronize());
                }) /
                iters; // cuda_time_ms 返回的是毫秒，所以这里直接除以迭代次数

    // 计算带宽（GB/s）
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double tflops = flops / (ms / 1000.0) / 1e12;

    // 计算 CPU 执行时间
    auto cpu_start_time = std::chrono::high_resolution_clock::now();

    // gemm_cpu(hA.data(), hB.data(), hA.data(), M, N, K); // 使用 CPU 计算

    auto cpu_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = cpu_end_time - cpu_start_time;
    double cpu_ms = cpu_duration.count() * 1000.0;
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
REGISTER_OP_FUNCS("gemm_demo3", correctness, perf);
