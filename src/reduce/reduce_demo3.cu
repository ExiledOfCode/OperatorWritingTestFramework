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
__global__ void reduce_demo3(const float *dA, float *dY, int M, int K) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    __shared__ float smem[256];
    float acc = 0.0f;
    for (int i = tx; i < K; i += blockDim.x)
        acc += dA[bx * K + i];

    smem[tx] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tx < stride)
            smem[tx] += smem[tx + stride];
        __syncthreads();
    }
    if (tx == 0)
        dY[bx] = smem[0];
}

static void reduce_hand(float *dY, const float *dA, int M, int K) {
    int threads = 256;
    dim3 block(threads);
    dim3 grid(M); // one block per row

    reduce_demo3<<<grid, block>>>(dA, dY, M, K);
    CUDA_CHECK(cudaGetLastError());
}

LC_REGISTER_REDUCE_ROW_OP("reduce_demo3", reduce_cpu_row_sum, reduce_hand);