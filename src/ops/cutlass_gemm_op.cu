#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <algorithm>

#include "utils.cuh"
#include "op_registry.hpp"

// CUTLASS
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"

static void gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int m=0;m<M;m++){
    for (int n=0;n<N;n++){
      float acc = 0.f;
      for (int k=0;k<K;k++){
        acc += A[m*K + k] * B[k*N + n];
      }
      C[m*N + n] = acc;
    }
  }
}

static void gemm_cutlass(float* dC, const float* dA, const float* dB, int M, int N, int K) {
  using Element = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using Gemm = cutlass::gemm::device::Gemm<
      Element, LayoutA,
      Element, LayoutB,
      Element, LayoutC,
      Element
  >;

  Element alpha = 1.0f;
  Element beta  = 0.0f;

  Gemm gemm_op;
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  typename Gemm::Arguments args(
      problem_size,
      {dA, K},
      {dB, N},
      {dC, N},
      {dC, N},
      {alpha, beta}
  );

  cutlass::Status status = gemm_op(args);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "CUTLASS GEMM failed, status = " << int(status) << "\n";
    std::exit(EXIT_FAILURE);
  }
}

// ======= 你要的：两个“启动函数” =======

static CorrectnessResult run_cutlass_gemm_f32_correctness() {
  int M=256, N=256, K=256;

  std::vector<float> hA((size_t)M*K), hB((size_t)K*N), hRef((size_t)M*N), hOut((size_t)M*N);

  std::mt19937 gen(123);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (auto& x: hA) x = dist(gen);
  for (auto& x: hB) x = dist(gen);

  gemm_cpu(hA.data(), hB.data(), hRef.data(), M, N, K);

  float *dA=nullptr, *dB=nullptr, *dC=nullptr;
  CUDA_CHECK(cudaMalloc(&dA, hA.size()*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, hB.size()*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, hOut.size()*sizeof(float)));

  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size()*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size()*sizeof(float), cudaMemcpyHostToDevice));

  gemm_cutlass(dC, dA, dB, M, N, K);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(hOut.data(), dC, hOut.size()*sizeof(float), cudaMemcpyDeviceToHost));

  float max_abs = 0.f;
  for (int i=0;i<M*N;i++){
    max_abs = std::max(max_abs, std::abs(hRef[i] - hOut[i]));
  }

  cudaFree(dA); cudaFree(dB); cudaFree(dC);

  double thr = 1e-2;
  return CorrectnessResult{max_abs <= thr, (double)max_abs, "max_abs_diff", thr, ""};
}

static PerfResult run_cutlass_gemm_f32_perf() {
  int M=1024, N=1024, K=1024;

  std::vector<float> hA((size_t)M*K, 1.f), hB((size_t)K*N, 1.f);

  float *dA=nullptr, *dB=nullptr, *dC=nullptr;
  CUDA_CHECK(cudaMalloc(&dA, hA.size()*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, hB.size()*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, (size_t)M*N*sizeof(float)));

  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size()*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size()*sizeof(float), cudaMemcpyHostToDevice));

  for(int i=0;i<10;i++) gemm_cutlass(dC, dA, dB, M, N, K);
  CUDA_CHECK(cudaDeviceSynchronize());

  int iters=50;
  float ms = cuda_time_ms([&](){
    for(int i=0;i<iters;i++) gemm_cutlass(dC, dA, dB, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
  }) / iters;

  double flops = 2.0 * (double)M * (double)N * (double)K;
  double tflops = flops / (ms/1000.0) / 1e12;

  cudaFree(dA); cudaFree(dB); cudaFree(dC);

  return PerfResult{(double)ms, "TFLOP/s", tflops, "FLOPs = 2*M*N*K"};
}

// ======= 一行注册（你想要的“宏包裹”） =======
REGISTER_OP_FUNCS("cutlass_gemm_f32", run_cutlass_gemm_f32_correctness, run_cutlass_gemm_f32_perf);
