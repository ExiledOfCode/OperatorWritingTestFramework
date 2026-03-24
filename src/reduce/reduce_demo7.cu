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
__device__ __forceinline__ void warp_reduce(volatile float *smem, const int &tx) {
    if (tx < 32) {
        smem[tx] += smem[tx + 32];
        smem[tx] += smem[tx + 16];
        smem[tx] += smem[tx + 8];
        smem[tx] += smem[tx + 4];
        smem[tx] += smem[tx + 2];
        smem[tx] += smem[tx + 1];
    }
}
__global__ void reduce_demo7(const float *dA, float *dY, int M, int K) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    volatile __shared__ float smem[256]; // 如果不加volatile会错误，因为编译器优化了smem的读取，会去寄存器里面读取
    float acc = 0.0f;
    for (int i = tx; i < K; i += blockDim.x)
        acc += dA[bx * K + i];

    smem[tx] = acc;
    __syncthreads();
#pragma unroll
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tx < stride)
            smem[tx] += smem[tx + stride];
        __syncthreads();
    }
    warp_reduce(smem, tx);
    if (tx == 0)
        dY[bx] = smem[0];
}

static void reduce_hand(float *dY, const float *dA, int M, int K) {
    int threads = 256;
    dim3 block(threads);
    dim3 grid(M); // one block per row

    reduce_demo7<<<grid, block>>>(dA, dY, M, K);
    CUDA_CHECK(cudaGetLastError());
}

LC_REGISTER_REDUCE_ROW_OP("reduce_demo7", reduce_cpu_row_sum, reduce_hand);