#include <algorithm>
#include <chrono> // 用于计时
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "op_registry.hpp"
#include "utils.cuh"

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
template<unsigned int thread_num_x,unsigned int thread_num_y>
__global__ void transpose_demo3(float *A, float *B, int M, int N) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x; // col
    int ty = blockIdx.y * blockDim.y + threadIdx.y; // row

    alignas(16) float src_reg[4][4];
    alignas(16) float dst_reg[4][4];

    for (int i = 0; i < 4; i++) {
        FETCH_FLOAT4(src_reg[i][0]) = FETCH_FLOAT4(A[(ty * thread_num_y + i) * N + (tx * thread_num_x)]);
    }

    for (int i = 0; i < 4; i++) {
        FETCH_FLOAT4(dst_reg[i][0]) = make_float4(src_reg[0][i], src_reg[1][i], src_reg[2][i], src_reg[3][i]);
    }

    for (int i = 0; i < 4; i++) {
        FETCH_FLOAT4(A[(tx * thread_num_x + i) * M + ty]) = FETCH_FLOAT4(dst_reg[i][0]);
    }
}

static void transpose_hand(float *dOut, float *dIn, int M, int N) {

    // 通过改变block的形状来达到合并访存的目的
    constexpr unsigned int block_x = 16;
    constexpr unsigned int block_y = 16;
    constexpr unsigned int num_threads_x = 4;
    constexpr unsigned int num_threads_y = 4;

    dim3 block1(block_x, block_y);
    dim3 grid1((N + (block_x * num_threads_x) - 1) / (block_x * num_threads_x), (M + (block_y * num_threads_y) - 1) / (block_y * num_threads_y));

    transpose_demo3<num_threads_x,num_threads_y><<<grid1, block1>>>(dIn, dOut, M, N);
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
REGISTER_OP_FUNCS("transpose_demo3", correctness, perf);