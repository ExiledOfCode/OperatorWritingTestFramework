#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <algorithm>

#include "utils.cuh"
#include "op_registry.hpp"

__global__ void vector_add_kernel(const float* a, const float* b, float* c, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) c[idx] = a[idx] + b[idx];
}

static void vector_add_cuda(const float* d_a, const float* d_b, float* d_c, size_t n) {
  int block = 256;
  int grid = (int)((n + block - 1) / block);
  vector_add_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
  CUDA_CHECK(cudaGetLastError());
}

static void vector_add_cpu(const float* a, const float* b, float* c, size_t n) {
  for (size_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

REGISTER_OP(
  "vector_add",
  ( []() -> CorrectnessResult {
    size_t n = 1 << 20;

    std::vector<float> h_a(n), h_b(n), h_ref(n), h_out(n);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (size_t i = 0; i < n; ++i) { h_a[i] = dist(gen); h_b[i] = dist(gen); }

    vector_add_cpu(h_a.data(), h_b.data(), h_ref.data(), n);

    float *d_a=nullptr, *d_b=nullptr, *d_c=nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, n*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n*sizeof(float), cudaMemcpyHostToDevice));

    vector_add_cuda(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_c, n*sizeof(float), cudaMemcpyDeviceToHost));

    float max_abs = 0.f;
    for (size_t i=0;i<n;i++){
      max_abs = std::max(max_abs, std::abs(h_ref[i] - h_out[i]));
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    double thr = 1e-5;
    return CorrectnessResult{max_abs <= thr, (double)max_abs, "max_abs_diff", thr, ""};
  } ),
  ( []() -> PerfResult {
    size_t n = 1 << 26;

    std::vector<float> h_a(n, 1.0f), h_b(n, 2.0f);
    float *d_a=nullptr, *d_b=nullptr, *d_c=nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, n*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n*sizeof(float), cudaMemcpyHostToDevice));

    for (int i=0;i<20;i++) vector_add_cuda(d_a,d_b,d_c,n);
    CUDA_CHECK(cudaDeviceSynchronize());

    int iters = 200;
    float ms = cuda_time_ms([&](){
      for(int i=0;i<iters;i++) vector_add_cuda(d_a,d_b,d_c,n);
      CUDA_CHECK(cudaDeviceSynchronize());
    }) / iters;

    double gbps = 3.0 * n * sizeof(float) / (ms/1000.0) / 1e9;

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return PerfResult{(double)ms, "GB/s", gbps, "bytes = read(A)+read(B)+write(C)"};
  } )
);
