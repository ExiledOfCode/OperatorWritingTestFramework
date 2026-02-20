#pragma once
#include <cstdlib>
#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>

#define CUDA_CHECK(call)                                                                                                                                       \
    do {                                                                                                                                                       \
        cudaError_t err = (call);                                                                                                                              \
        if (err != cudaSuccess) {                                                                                                                              \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl;                                      \
            std::exit(EXIT_FAILURE);                                                                                                                           \
        }                                                                                                                                                      \
    } while (0)

inline float cuda_time_ms(const std::function<void()> &func) {
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

inline void dump_matrix_csv(std::ofstream &ofs, const std::string &name, const float *data, int rows, int cols) {
    ofs << name << "\n";
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            ofs << std::setprecision(8) << data[r * cols + c];
            if (c != cols - 1)
                ofs << ",";
        }
        ofs << "\n";
    }
    ofs << "\n";
}

inline void dump_to_csv(const std::string &filename, const float *A, const float *B, const float *C_gpu, const float *C_cpu, int M, int N, int K) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    ofs << std::fixed;
    // A: M x K
    dump_matrix_csv(ofs, "Matrix A (MxK)", A, M, K);
    // B: K x N
    dump_matrix_csv(ofs, "Matrix B (KxN)", B, K, N);
    // C_gpu: M x N
    dump_matrix_csv(ofs, "Matrix C_gpu (MxN)", C_gpu, M, N);
    // C_cpu: M x N
    dump_matrix_csv(ofs, "Matrix C_cpu (MxN)", C_cpu, M, N);
    ofs.close();
    std::cout << "Dumped matrices to " << filename << std::endl;
}