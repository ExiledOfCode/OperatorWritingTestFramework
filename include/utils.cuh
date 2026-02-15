#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <functional>
#include <cstdlib>

#define CUDA_CHECK(call)                                      \
  do {                                                        \
    cudaError_t err = (call);                                  \
    if (err != cudaSuccess) {                                 \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err)  \
                << " at " << __FILE__ << ":" << __LINE__      \
                << std::endl;                                 \
      std::exit(EXIT_FAILURE);                                \
    }                                                         \
  } while (0)

inline float cuda_time_ms(const std::function<void()>& func) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  func();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms;
}
