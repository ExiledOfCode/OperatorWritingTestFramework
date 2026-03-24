#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "op_registry.hpp"
#include "op_runtime.hpp"
#include "utils.cuh"

__global__ void vector_add_kernel(const float *a, const float *b, float *c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

static void vector_add_cuda(const float *d_a, const float *d_b, float *d_c, size_t n) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    vector_add_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
}

// static void vector_add_cpu(const float *a, const float *b, float *c, size_t n) {
//     for (size_t i = 0; i < n; ++i)
//         c[i] = a[i] + b[i];
// }

static void vector_add_launch(const float *d_a, const float *d_b, float *d_c, size_t n) {
    vector_add_cuda(d_a, d_b, d_c, n);
}

LC_REGISTER_VECTOR_BINARY_OP("vector_add", opfw::cpu_vector_add_f32, vector_add_launch);
