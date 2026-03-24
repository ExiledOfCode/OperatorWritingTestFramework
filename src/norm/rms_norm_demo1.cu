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

static void rms_norm_cpu(const float *A, float *Y, int M, int K) {
    for (int m = 0; m < M; ++m) {
        const float *row_A = A + static_cast<size_t>(m) * K;
        float *row_Y = Y + static_cast<size_t>(m) * K;

        float sumsq = 0.0f;
        for (int k = 0; k < K; ++k) {
            const float v = row_A[k];
            sumsq += v * v;
        }

        const float inv_rms = 1.0f / std::sqrt(sumsq / static_cast<float>(K) + kEpsilon);
        for (int k = 0; k < K; ++k) {
            row_Y[k] = row_A[k] * inv_rms;
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

__global__ void rms_norm_demo1(const float *dA, float *dY, int M, int K) {
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int lane = tx & 31;
    const int wid = tx >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;

    __shared__ float smem_sumsq[32];
    __shared__ float inv_rms;

    float sumsq = 0.0f;
    const float *row_A = dA + static_cast<size_t>(bx) * K;
    float *row_Y = dY + static_cast<size_t>(bx) * K;

    for (int i = tx; i < K; i += blockDim.x) {
        const float v = row_A[i];
        sumsq += v * v;
    }

    sumsq = warp_reduce_sum(sumsq);

    if (lane == 0) {
        smem_sumsq[wid] = sumsq;
    }
    __syncthreads();

    if (wid == 0) {
        float block_sumsq = (lane < num_warps) ? smem_sumsq[lane] : 0.0f;
        block_sumsq = warp_reduce_sum(block_sumsq);

        if (lane == 0) {
            inv_rms = rsqrtf(block_sumsq / static_cast<float>(K) + kEpsilon);
        }
    }
    __syncthreads();

    for (int i = tx; i < K; i += blockDim.x) {
        row_Y[i] = row_A[i] * inv_rms;
    }
}

static void rms_norm_hand(float *dY, const float *dA, int M, int K) {
    const int threads = 256;
    dim3 block(threads);
    dim3 grid(M);

    rms_norm_demo1<<<grid, block>>>(dA, dY, M, K);
    CUDA_CHECK(cudaGetLastError());
}

static opfw::NormRowSpec rms_norm_spec() {
    opfw::NormRowSpec spec;
    spec.shape = {1024, 4096};
    spec.correctness_max_shape = {1024, 4096};
    spec.threshold = 1e-4;
    spec.description = "Y[M,K] = rms_norm(X[M,K]), eps=1e-5";
    spec.note = "bytes = read(X)+write(Y)";
    return spec;
}

} // namespace

LC_REGISTER_NORM_ROW_OP_EX("rms_norm_demo1", rms_norm_cpu, rms_norm_hand, rms_norm_spec());
