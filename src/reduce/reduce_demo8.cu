#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "op_registry.hpp"
#include "op_runtime.hpp"
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

LC_REGISTER_REDUCE_ROW_OP("reduce_demo8", reduce_cpu_row_sum, reduce_hand);