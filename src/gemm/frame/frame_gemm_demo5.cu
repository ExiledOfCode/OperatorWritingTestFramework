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
// shared memory:
// read:
// write:   MN
// global memory:
// read:     (M/bm)*(N/bn)个块，每个块(bm*K+K*bn)次访存，总计：KMN(1/bm+1/bn)次
// write:    MN
template <typename T>
struct MatrixView {
    T *ptr;
    int width;
    __device__ __forceinline__ T &operator()(int r, int c) { return ptr[r * width + c]; }
};

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

template <unsigned int N_NUM_PRE_BLOCK, unsigned int M_NUM_PER_BLOCK, unsigned int K_NUM_PER_BLOCK, unsigned int NUM_PER_THREAD, unsigned int REG_PER_THREAD>
__global__ void frame_gemm_demo5(float *dA, float *dB, float *dC, int M, int N, int K) {

    MatrixView<float> A{dA, K};
    MatrixView<float> B{dB, N};
    MatrixView<float> C{dC, N};

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int block_base_x = (blockDim.x * NUM_PER_THREAD) * blockIdx.x;
    int block_base_y = blockDim.y * blockIdx.y;

    // int col = tx + block_base_x;
    int row = ty + block_base_y;

    // 二维索引的映射变换
    int block_idx = tx + blockDim.x * ty;

    int reg_tx = block_idx % (K_NUM_PER_BLOCK / REG_PER_THREAD);
    int reg_ty = block_idx / (K_NUM_PER_BLOCK / REG_PER_THREAD);

    __shared__ float A_shared[N_NUM_PRE_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float B_shared[K_NUM_PER_BLOCK][N_NUM_PRE_BLOCK];

    float a_reg[REG_PER_THREAD] = {0.0f};
    float b_reg[REG_PER_THREAD] = {0.0f};
    float tmp[REG_PER_THREAD * REG_PER_THREAD] = {0.0f};

    for (int s = 0; s < K / K_NUM_PER_BLOCK; s++) {
        const int k_base = s * K_NUM_PER_BLOCK;
        FETCH_FLOAT4(A_shared[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(A(row, tx * NUM_PER_THREAD + k_base));
        FETCH_FLOAT4(B_shared[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(B(ty + k_base, block_base_x + NUM_PER_THREAD * tx));

        __syncthreads();

        for (int k = 0; k < K_NUM_PER_BLOCK; k++) {
            a_reg[0] = A_shared[reg_ty * 2][k];
            a_reg[1] = A_shared[reg_ty * 2 + 1][k];
            b_reg[0] = B_shared[k][reg_tx * 2];
            b_reg[1] = B_shared[k][reg_tx * 2 + 1];
            for (int i = 0; i < REG_PER_THREAD; i++) {
                for (int j = 0; j < REG_PER_THREAD; j++) {
                    tmp[i * REG_PER_THREAD + j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < REG_PER_THREAD; i++) {
        for (int j = 0; j < REG_PER_THREAD; j++) {
            C(block_base_y + reg_ty * 2 + i, block_base_x + reg_tx * 2 + j) = tmp[i * REG_PER_THREAD + j];
        }
    }
}

// CUTLASS GPU 实现
static void gemm_hand(float *dC, float *dA, float *dB, int M, int N, int K) {

    constexpr int BLOCK_SIZE = 16;
    constexpr int STRIDE = 1;
    constexpr int TILE = BLOCK_SIZE * STRIDE;

    constexpr int N_NUM_PRE_BLOCK = 16;
    constexpr int M_NUM_PER_BLOCK = 16;
    constexpr int K_NUM_PER_BLOCK = 16;
    constexpr int NUM_PER_THREAD = 4;

    constexpr int REG_PER_THREAD = 2;

    dim3 block(M_NUM_PER_BLOCK / NUM_PER_THREAD, N_NUM_PRE_BLOCK);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    frame_gemm_demo5<N_NUM_PRE_BLOCK, M_NUM_PER_BLOCK, K_NUM_PER_BLOCK, NUM_PER_THREAD, REG_PER_THREAD><<<grid, block>>>(dA, dB, dC, M, N, K);
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
REGISTER_OP_FUNCS("frame_gemm_demo5", correctness, perf);
