#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "op_registry.hpp"
#include "op_runtime.hpp"
#include "utils.cuh"

namespace {

constexpr float kEpsilon = 1.0e-5f;

static void layer_norm_cpu(const float *A, float *Y, int M, int K) {
    for (int m = 0; m < M; ++m) {
        const float *row_A = A + static_cast<size_t>(m) * K;
        float *row_Y = Y + static_cast<size_t>(m) * K;

        float sum = 0.0f;
        float sumsq = 0.0f;
        for (int k = 0; k < K; ++k) {
            const float v = row_A[k];
            sum += v;
            sumsq += v * v;
        }

        const float mean = sum / static_cast<float>(K);
        const float var = sumsq / static_cast<float>(K) - mean * mean;
        const float inv_std = 1.0f / std::sqrt(var + kEpsilon);

        for (int k = 0; k < K; ++k) {
            row_Y[k] = (row_A[k] - mean) * inv_std;
        }
    }
}

__device__ __forceinline__ float warp_reduce_sum(float value, unsigned mask = 0xffffffffu) {
    value += __shfl_down_sync(mask, value, 16);
    value += __shfl_down_sync(mask, value, 8);
    value += __shfl_down_sync(mask, value, 4);
    value += __shfl_down_sync(mask, value, 2);
    value += __shfl_down_sync(mask, value, 1);
    return value;
}

__global__ void layer_norm_demo1(const float *dA, float *dY, int M, int K) {
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int lane = tx & 31;
    const int wid = tx >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;

    __shared__ float smem_sum[32];
    __shared__ float smem_sumsq[32];
    __shared__ float mean;
    __shared__ float inv_std;

    float sum = 0.0f;
    float sumsq = 0.0f;
    const float *row_A = dA + static_cast<size_t>(bx) * K;
    float *row_Y = dY + static_cast<size_t>(bx) * K;

    for (int i = tx; i < K; i += blockDim.x) {
        const float v = row_A[i];
        sum += v;
        sumsq += v * v;
    }

    sum = warp_reduce_sum(sum);
    sumsq = warp_reduce_sum(sumsq);

    if (lane == 0) {
        smem_sum[wid] = sum;
        smem_sumsq[wid] = sumsq;
    }
    __syncthreads();

    if (wid == 0) {
        float block_sum = (lane < num_warps) ? smem_sum[lane] : 0.0f;
        float block_sumsq = (lane < num_warps) ? smem_sumsq[lane] : 0.0f;

        block_sum = warp_reduce_sum(block_sum);
        block_sumsq = warp_reduce_sum(block_sumsq);

        if (lane == 0) {
            mean = block_sum / static_cast<float>(K);
            const float var = block_sumsq / static_cast<float>(K) - mean * mean;
            inv_std = rsqrtf(var + kEpsilon);
        }
    }
    __syncthreads();

    for (int i = tx; i < K; i += blockDim.x) {
        row_Y[i] = (row_A[i] - mean) * inv_std;
    }
}

static void layer_norm_hand(float *dY, const float *dA, int M, int K) {
    const int threads = 256;
    dim3 block(threads);
    dim3 grid(M);

    layer_norm_demo1<<<grid, block>>>(dA, dY, M, K);
    CUDA_CHECK(cudaGetLastError());
}

static opfw::NormRowSpec layer_norm_spec() {
    opfw::NormRowSpec spec;
    spec.shape = {1024, 4096};
    spec.correctness_max_shape = {1024, 4096};
    spec.threshold = 1e-4;
    spec.description = "Y[M,K] = layer_norm(X[M,K]), eps=1e-5";
    spec.note = "bytes = read(X)+write(Y)";
    return spec;
}

} // namespace

LC_REGISTER_NORM_ROW_OP_EX("layer_norm_demo1", layer_norm_cpu, layer_norm_hand, layer_norm_spec());
