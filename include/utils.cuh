#pragma once
#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

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
            ofs << std::setprecision(4) << data[r * cols + c];
            if (c != cols - 1)
                ofs << ",";
        }
        ofs << "\n";
    }
    ofs << "\n";
}

static size_t numel(const std::vector<int> &shape) {
    if (shape.empty())
        return 0;
    size_t t = 1;
    for (int d : shape)
        t *= static_cast<size_t>(d);
    return t;
}

static void dump_tensor_csv(std::ofstream &ofs, const std::string &name, const float *data, const std::vector<int> &shape) {
    ofs << std::fixed << std::setprecision(3);
    ofs << "==== " << name << " ====\n";
    ofs << "shape,";
    for (size_t i = 0; i < shape.size(); ++i) {
        ofs << shape[i] << (i + 1 < shape.size() ? "x" : "");
    }
    ofs << "\n";

    if (!data) {
        ofs << "ERROR,null data pointer\n\n";
        return;
    }
    if (shape.empty()) {
        ofs << "ERROR,empty shape\n\n";
        return;
    }

    const size_t ndim = shape.size();

    // 1D: 输出一行
    if (ndim == 1) {
        int L = shape[0];
        for (int i = 0; i < L; ++i) {
            ofs << data[i] << (i + 1 < L ? "," : "");
        }
        ofs << "\n\n";
        return;
    }

    // 2D: 正常矩阵
    auto dump_2d = [&](const float *base, int rows, int cols) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                ofs << base[r * cols + c] << (c + 1 < cols ? "," : "");
            }
            ofs << "\n";
        }
    };

    int rows = shape[ndim - 2];
    int cols = shape[ndim - 1];

    if (ndim == 2) {
        dump_2d(data, rows, cols);
        ofs << "\n";
        return;
    }

    // >=3D: 前面维度 flatten 成 batch，逐块输出 2D
    size_t batch = 1;
    for (size_t i = 0; i + 2 < ndim; ++i)
        batch *= static_cast<size_t>(shape[i]);
    size_t stride2d = static_cast<size_t>(rows) * static_cast<size_t>(cols);

    for (size_t b = 0; b < batch; ++b) {
        ofs << "-- slice " << b << " / " << batch << "\n";
        dump_2d(data + b * stride2d, rows, cols);
        ofs << "\n";
    }
}

struct DumpItem {
    std::string name;
    const float *ptr;
    std::vector<int> shape;
};

inline void dump_to_csv_any(const std::string &filename, const std::vector<DumpItem> &items) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    ofs << std::fixed << std::setprecision(6);

    for (const auto &it : items) {
        dump_tensor_csv(ofs, it.name, it.ptr, it.shape);
    }

    ofs.close();
    std::cout << "Dumped tensors to " << filename << std::endl;
}