#include <algorithm>
#include <chrono> // 用于计时
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "op_registry.hpp"
#include "op_runtime.hpp"
#include "utils.cuh"

// 参考 CPU 实现
// static void gemm_cpu(const float *A, const float *B, float *C, int M, int N, int K) {
//     for (int m = 0; m < M; m++) {
//         for (int n = 0; n < N; n++) {
//             float acc = 0.f;
//             for (int k = 0; k < K; k++) {
//                 acc += A[m * K + k] * B[k * N + n];
//             }
//             C[m * N + n] = acc;
//         }
//     }
// }
// 29.2170887ms
// shared memory:
// read:
// write:
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
template <int BLOCK_SIZE, int STRIDE = 1>
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

template <unsigned int BLOCK_SIZE>
__global__ void frame_gemm_demo2(const float *dA, const float *dB, float *dC, int M, int N, int K) {

    MatrixView<float> A{dA, K};
    MatrixView<float> B{dB, N};
    MatrixViewMut<float> C{dC, N};

    GemmMap<BLOCK_SIZE> map;
    extern __shared__ float smem[];

    SharedTile<BLOCK_SIZE, float> shared_A{smem};
    SharedTile<BLOCK_SIZE, float> shared_B{smem + BLOCK_SIZE * BLOCK_SIZE};

    float tmp = 0.0f;

    for (int s = 0; s < K / BLOCK_SIZE; s++) {
        int k_base = s * BLOCK_SIZE;

        shared_A(map.window2shared_row(0), map.window2shared_col(0)) = A(map.window2grid_row(0), k_base + map.window2shared_col(0));
        shared_B(map.window2shared_row(0), map.window2shared_col(0)) = B(k_base + map.window2shared_row(0), map.window2grid_col(0));

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; i++) {
            tmp += shared_A(map.window2shared_row(0), i) * shared_B(i, map.window2shared_col(0));
        }

        __syncthreads();
    }
    C(map.thread2grid_row(), map.thread2grid_col()) = tmp;
}

// CUTLASS GPU 实现
static void gemm_hand(float *dC, const float *dA, const float *dB, int M, int N, int K) {

    dim3 block(16, 16);
    // grid 按输出矩阵 C 的 (M,N) 覆盖
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    int shared_size = 2 * 16 * 16 * sizeof(float);
    frame_gemm_demo2<16><<<grid, block, shared_size>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

LC_REGISTER_GEMM_OP("frame_gemm_demo2", opfw::cpu_gemm_f32, gemm_hand);
