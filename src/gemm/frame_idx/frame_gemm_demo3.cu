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
template <typename T>
struct MatrixView {
    const T *ptr;
    int width; // leading dimension
    __device__ __forceinline__ T operator()(int r, int c) const {
        // (r,c)->idx
        return ptr[r * width + c];
    }
};

template <typename T>
struct MatrixViewMut {
    T *ptr;
    int width;
    __device__ __forceinline__ T &operator()(int r, int c) {
        // (r,c)->idx
        return ptr[r * width + c];
    }
};

template <int WIDTH, typename T>
struct SharedTile {
    T *ptr;
    __device__ __forceinline__ T &operator()(int r, int c) {
        // (r,c)->idx
        return ptr[r * WIDTH + c];
    }
    __device__ __forceinline__ T operator()(int r, int c) const { return ptr[r * WIDTH + c]; }
};

// block 管 TILE×TILE；thread 负责 STRIDE×STRIDE 的 window(micro-tile)
template <int BLOCK_SIZE, int STRIDE>
struct GemmMap {
    static constexpr int TILE = BLOCK_SIZE * STRIDE;

    int block_row_base; // C tile 左上角全局行
    int block_col_base; // C tile 左上角全局列
    int tx, ty;         // thread 坐标 (x,y)

    __device__ __forceinline__ GemmMap()
        : block_row_base(int(blockIdx.y) * TILE), block_col_base(int(blockIdx.x) * TILE), tx(int(threadIdx.x)), ty(int(threadIdx.y)) {}

    // thread (tx,ty) 对应的 window 左上角（其实就是它负责的第一个点）
    __device__ __forceinline__ int thread2grid_row() const { return block_row_base + ty; }
    __device__ __forceinline__ int thread2grid_col() const { return block_col_base + tx; }

    // window -> tile(shared) 坐标映射
    __device__ __forceinline__ int window2shared_row(int y_stride) const { return ty + y_stride * BLOCK_SIZE; }
    __device__ __forceinline__ int window2shared_col(int x_stride) const { return tx + x_stride * BLOCK_SIZE; }

    // window -> grid(global) 坐标映射
    __device__ __forceinline__ int window2grid_row(int y_stride) const { return block_row_base + window2shared_row(y_stride); }
    __device__ __forceinline__ int window2grid_col(int x_stride) const { return block_col_base + window2shared_col(x_stride); }
};

template <unsigned int BLOCK_SIZE, unsigned int STRIDE>
__global__ void frame_gemm_demo3(const float *__restrict__ dA, const float *__restrict__ dB, float *__restrict__ dC, int M, int N, int K) {
    constexpr int TILE = BLOCK_SIZE * STRIDE;

    MatrixView<float> A{dA, K};    // A: MxK
    MatrixView<float> B{dB, N};    // B: KxN
    MatrixViewMut<float> C{dC, N}; // C: MxN

    GemmMap<BLOCK_SIZE, STRIDE> map;

    // 只算完整 tile（保持你原版行为）
    if (map.block_row_base + TILE > M || map.block_col_base + TILE > N)
        return;

    extern __shared__ float smem[];
    SharedTile<TILE, float> shared_A{smem};
    SharedTile<TILE, float> shared_B{smem + TILE * TILE};

    float tC[STRIDE][STRIDE] = {0.0f};

    const int num_k_tiles = K / TILE;

    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int k_base = k_tile * TILE;
#pragma unroll
        for (int y_stride = 0; y_stride < (int)STRIDE; ++y_stride) {
            for (int x_stride = 0; x_stride < (int)STRIDE; ++x_stride) {
                const int tile_r = map.window2shared_row(y_stride);
                const int tile_c = map.window2shared_col(x_stride);

                // A -> shared_A
                shared_A(tile_r, tile_c) = A(map.window2grid_row(y_stride), k_base + tile_c);

                // B -> shared_B
                shared_B(tile_r, tile_c) = B(k_base + tile_r, map.window2grid_col(x_stride));
            }
        }
        __syncthreads();

#pragma unroll
        for (int y_stride = 0; y_stride < (int)STRIDE; ++y_stride) {
            for (int x_stride = 0; x_stride < (int)STRIDE; ++x_stride) {
                const int tile_r = map.window2shared_row(y_stride);
                const int tile_c = map.window2shared_col(x_stride);

                float acc = tC[y_stride][x_stride];
                for (int i = 0; i < TILE; ++i) {
                    acc += shared_A(tile_r, i) * shared_B(i, tile_c);
                }
                tC[y_stride][x_stride] = acc;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int y_stride = 0; y_stride < (int)STRIDE; ++y_stride) {
        for (int x_stride = 0; x_stride < (int)STRIDE; ++x_stride) {
            C(map.window2grid_row(y_stride), map.window2grid_col(x_stride)) = tC[y_stride][x_stride];
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

    frame_gemm_demo3<16, STRIDE><<<grid, block, shared_size>>>(dA, dB, dC, M, N, K);
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
REGISTER_OP_FUNCS("frame_gemm_demo3", correctness, perf);
