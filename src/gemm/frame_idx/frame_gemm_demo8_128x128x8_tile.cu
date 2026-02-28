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

template <unsigned int BLOCK_TILE_SIZE_M, unsigned int BLOCK_TILE_SIZE_N, unsigned int BLOCK_TILE_SIZE_K, unsigned int THREAD_NUM_X, unsigned int THREAD_NUM_Y>
__global__ void frame_gemm_demo8_128x128x8_tile(float *dA, float *dB, float *dC, int M, int N, int K) {

    MatrixView<float> A{dA, K};
    MatrixView<float> B{dB, N};
    MatrixView<float> C{dC, N};

    __shared__ float shared_A[2][BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_M];
    __shared__ float shared_B[2][BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_N];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int block_tile_C_start_x = bx * (blockDim.x * THREAD_NUM_X);
    int block_tile_C_start_y = by * (blockDim.y * THREAD_NUM_Y);

    int thread_tile_C_start_x = block_tile_C_start_x + tx * THREAD_NUM_X;
    int thread_tile_C_start_y = block_tile_C_start_y + ty * THREAD_NUM_Y;

    int thread_shared_tile_A_start_y = ty * THREAD_NUM_Y;
    int thread_shared_tile_B_start_x = tx * THREAD_NUM_X;

    alignas(16) float reg_A[THREAD_NUM_Y] = {0.0f};
    alignas(16) float reg_B[THREAD_NUM_X] = {0.0f};

    float tmp[THREAD_NUM_Y][THREAD_NUM_X] = {0.0f};
    float *reg_load_shared_A = reg_A;

    int cnt = 0;
    for (int j = 0; j < THREAD_NUM_Y; j++) {
        for (int i = 0; i < THREAD_NUM_X; i += 4) {
            FETCH_FLOAT4(reg_load_shared_A[i]) = FETCH_FLOAT4(A(thread_tile_C_start_y + j, i));
        }
        for (int i = 0; i < THREAD_NUM_X; i++) {
            shared_A[0][i][thread_shared_tile_A_start_y + j] = reg_load_shared_A[i];
        }
    }

    for (int j = 0; j < THREAD_NUM_Y; j++) {
        for (int i = 0; i < THREAD_NUM_X; i += 4) {
            FETCH_FLOAT4(shared_B[0][j][thread_shared_tile_B_start_x + i]) = FETCH_FLOAT4(B(j, thread_tile_C_start_x + i));
        }
    }

    __syncthreads();

    for (int k_base = BLOCK_TILE_SIZE_K; k_base < K; k_base += BLOCK_TILE_SIZE_K) {
        for (int j = 0; j < THREAD_NUM_Y; j++) {
            for (int i = 0; i < THREAD_NUM_X; i += 4) {
                FETCH_FLOAT4(reg_load_shared_A[i]) = FETCH_FLOAT4(A(thread_tile_C_start_y + j, k_base + i));
            }
            for (int i = 0; i < THREAD_NUM_X; i++) {
                shared_A[cnt ^ 1][i][thread_shared_tile_A_start_y + j] = reg_load_shared_A[i];
            }
        }

        for (int j = 0; j < THREAD_NUM_Y; j++) {
            for (int i = 0; i < THREAD_NUM_X; i += 4) {
                FETCH_FLOAT4(shared_B[cnt ^ 1][j][thread_shared_tile_B_start_x + i]) = FETCH_FLOAT4(B(k_base + j, thread_tile_C_start_x + i));
            }
        }

        for (int k = 0; k < BLOCK_TILE_SIZE_K; k++) {
            for (int i = 0; i < THREAD_NUM_Y; i += 4) {
                FETCH_FLOAT4(reg_A[i]) = FETCH_FLOAT4(shared_A[cnt][k][thread_shared_tile_A_start_y + i]);
            }
            for (int i = 0; i < THREAD_NUM_X; i += 4) {
                FETCH_FLOAT4(reg_B[i]) = FETCH_FLOAT4(shared_B[cnt][k][thread_shared_tile_B_start_x + i]);
            }
            for (int i = 0; i < THREAD_NUM_Y; i++) {
                for (int j = 0; j < THREAD_NUM_X; j++) {
                    tmp[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }
        cnt ^= 1;
        __syncthreads();
    }
    for (int k = 0; k < BLOCK_TILE_SIZE_K; k++) {
        for (int i = 0; i < THREAD_NUM_Y; i += 4) {
            FETCH_FLOAT4(reg_A[i]) = FETCH_FLOAT4(shared_A[cnt][k][thread_shared_tile_A_start_y + i]);
        }
        for (int i = 0; i < THREAD_NUM_X; i += 4) {
            FETCH_FLOAT4(reg_B[i]) = FETCH_FLOAT4(shared_B[cnt][k][thread_shared_tile_B_start_x + i]);
        }
        for (int i = 0; i < THREAD_NUM_Y; i++) {
            for (int j = 0; j < THREAD_NUM_X; j++) {
                tmp[i][j] += reg_A[i] * reg_B[j];
            }
        }
    }
    __syncthreads();

    for (int i = 0; i < THREAD_NUM_Y; i++) {
        for (int j = 0; j < THREAD_NUM_X; j += 4) {
            FETCH_FLOAT4(C(thread_tile_C_start_y + i, thread_tile_C_start_x + j)) = FETCH_FLOAT4(tmp[i][j]);
        }
    }
}

// CUTLASS GPU 实现
static void gemm_hand(float *dC, float *dA, float *dB, int M, int N, int K) {
    constexpr int BLOCK_TILE_SIZE_M = 128;
    constexpr int BLOCK_TILE_SIZE_N = 128;
    constexpr int BLOCK_TILE_SIZE_K = 8;

    constexpr int THREAD_SIZE_X = 8;
    constexpr int THREAD_SIZE_Y = 8;

    dim3 block(BLOCK_TILE_SIZE_N / THREAD_SIZE_X, BLOCK_TILE_SIZE_M / THREAD_SIZE_Y);
    dim3 grid((N + BLOCK_TILE_SIZE_N - 1) / BLOCK_TILE_SIZE_N, (M + BLOCK_TILE_SIZE_M - 1) / BLOCK_TILE_SIZE_M);

    frame_gemm_demo8_128x128x8_tile<BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_K, THREAD_SIZE_X, THREAD_SIZE_Y>
        <<<grid, block>>>(dA, dB, dC, M, N, K);

    CUDA_CHECK(cudaGetLastError());
}

// ======= 你要的：两个“启动函数” =======

static CorrectnessResult correctness() {
    int M = 128, N = 128, K = 128;

    std::vector<float> hA((size_t)M * K), hB((size_t)K * N), hRef((size_t)M * N), hOut((size_t)M * N);

    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto &x : hA)
        x = abs(dist(gen));
    for (auto &x : hB)
        x = abs(dist(gen));

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

    // dump_to_csv_any("gemm_dump.csv", {
    //                                      {"A", hA.data(), {M, K}},
    //                                      {"B", hB.data(), {K, N}},
    //                                      {"C_gpu", hOut.data(), {M, N}},
    //                                      {"C_cpu", hRef.data(), {M, N}},
    //                                  });

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

    int iters = 1;
    // 预热：进行一些内核调用
    for (int i = 0; i < iters; i++)
        gemm_hand(dC, dA, dB, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 开始计时：使用 cuda_time_ms 测量 GPU 执行时间
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
REGISTER_OP_FUNCS("frame_gemm_demo8_128x128x8_tile", correctness, perf);
