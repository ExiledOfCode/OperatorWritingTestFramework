#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "op_registry.hpp"
#include "utils.cuh"

// =======================
// CPU reference: row-wise reduce
// input: A [M, K]
// output: Y [M]
// =======================
static void reduce_cpu_row_sum(const float *A, float *Y, int M, int K) {
    for (int m = 0; m < M; m++) {
        float acc = 0.f;
        for (int k = 0; k < K; k++) {
            acc += A[m * K + k];
        }
        Y[m] = acc;
    }
}

// =======================
// CUDA kernel: row-wise reduce
// One block handles one row
// =======================
__device__ __forceinline__ float warp_reduce_sum(float v, unsigned mask = 0xffffffffu) {
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

__global__ void reduce_demo8(const float *dA, float *dY, int M, int K) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    int lane = tx & 31;
    int wid = tx >> 5;
    int num_warps = (blockDim.x + 31) >> 5;

    __shared__ float smem[32];

    float acc = 0.0f;
    for (int i = tx; i < K; i += blockDim.x)
        acc += dA[bx * K + i];

    acc = warp_reduce_sum(acc);

    if (lane == 0)
        smem[wid] = acc;
    __syncthreads();

    if (wid == 0) {
        float v = (lane < num_warps) ? smem[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0)
            dY[bx] = v;
    }
}

static void reduce_hand(float *dY, const float *dA, int M, int K) {
    int threads = 256;
    dim3 block(threads);
    dim3 grid(M); // one block per row

    reduce_demo8<<<grid, block>>>(dA, dY, M, K);
    CUDA_CHECK(cudaGetLastError());
}

// ======= correctness =======
static CorrectnessResult correctness() {
    int M = 64, K = 4096;

    std::vector<float> hA((size_t)M * K), hRef((size_t)M), hOut((size_t)M);

    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto &x : hA)
        x = dist(gen);

    reduce_cpu_row_sum(hA.data(), hRef.data(), M, K);

    float *dA = nullptr, *dY = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dY, hOut.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));

    reduce_hand(dY, dA, M, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hOut.data(), dY, hOut.size() * sizeof(float), cudaMemcpyDeviceToHost));

    float max_abs = 0.f;
    for (int i = 0; i < M; i++) {
        max_abs = std::max(max_abs, std::abs(hRef[i] - hOut[i]));
    }

    // dump_to_csv_any("reduce_dump.csv", {
    //     {"hRef", hRef.data(), {M}},
    //     {"hOut", hOut.data(), {M}},
    //     {"max_abs_error", &max_abs, {1}},
    // });

    cudaFree(dA);
    cudaFree(dY);

    double thr = 1e-2;
    return CorrectnessResult{max_abs <= thr, (double)max_abs, "max_abs_diff", thr, "reduce row-sum: A[M,K]->Y[M]"};
}

// ======= perf =======
static PerfResult perf() {
    size_t M = 1 << 16; // 65536 rows
    size_t K = 10240;   // reduce length
    std::vector<float> hA((size_t)M * K, 1.f);

    float *dA = nullptr, *dY = nullptr;

    size_t input_size = (size_t)M * K * sizeof(float);
    size_t output_size = (size_t)M * sizeof(float);

    CUDA_CHECK(cudaMalloc(&dA, input_size));
    CUDA_CHECK(cudaMalloc(&dY, output_size));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), input_size, cudaMemcpyHostToDevice));

    // warmup
    for (int i = 0; i < 3; i++)
        reduce_hand(dY, dA, (int)M, (int)K);
    CUDA_CHECK(cudaDeviceSynchronize());

    int iters = 20;
    double ms = cuda_time_ms([&]() {
                    for (int i = 0; i < iters; i++)
                        reduce_hand(dY, dA, (int)M, (int)K);
                    CUDA_CHECK(cudaDeviceSynchronize());
                }) /
                iters;

    // 简单吞吐：读取 M*K 个 float，写 M 个 float
    // 这里只给一个“effective bandwidth”口径（不算缓存命中等）
    double bytes = (double)input_size + (double)output_size;
    double gbps = bytes / (ms / 1000.0) / 1e9;

    // CPU time（这里给个空壳，和你原来保持接口一致）
    auto cpu_start_time = std::chrono::high_resolution_clock::now();
    auto cpu_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = cpu_end_time - cpu_start_time;
    double cpu_ms = cpu_duration.count() * 1000.0;

    cudaFree(dA);
    cudaFree(dY);

    return PerfResult{(double)ms, cpu_ms, "GB/s", gbps, "Bytes = M*K*4 (read) + M*4 (write)", input_size, output_size, {{M, K}}, {{M}}};
}

// ======= register =======
REGISTER_OP_FUNCS("reduce_demo8", correctness, perf);