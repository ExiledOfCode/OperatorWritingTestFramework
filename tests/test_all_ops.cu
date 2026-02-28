#include "op_registry.hpp"
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_set>

// 获取要测试的算子集合
std::unordered_set<std::string> get_tested_operators(const std::string &filename) {
    std::unordered_set<std::string> operators;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return operators;
    }

    std::string op_name;
    while (std::getline(file, op_name)) {
        if (!op_name.empty() && op_name[0] != '#') {
            operators.insert(op_name); // 将每行算子名插入到集合中
        }
    }
    file.close();
    return operators;
}

TEST(AllOps, Correctness) {
    auto &reg = op_registry();
    ASSERT_FALSE(reg.empty()) << "No ops registered. Did you forget REGISTER_OP?";

    // 读取文件中的算子名
    std::unordered_set<std::string> operators_to_test = get_tested_operators("./test_op_name.sh");
    for (auto &e : reg) {
        if (operators_to_test.empty() || operators_to_test.find(e.name) != operators_to_test.end()) {
            SCOPED_TRACE("Op: " + e.name);
            auto r = e.correctness();
            bool pass = r.ok && (r.metric <= r.threshold);
            if (pass) {
                std::cout << "\033[1;32m" << "[AC] " << "\033[0m" << e.name << "  " << r.metric_name << "=" << std::setprecision(10) << r.metric
                          << " (thr=" << r.threshold << ")" << (r.note.empty() ? "" : ("  " + r.note)) << "\n";
            } else {
                std::cout << "\033[1;31m" << "[WA] " << "\033[0m" << e.name << "  " << r.metric_name << "=" << std::setprecision(10) << r.metric
                          << " (thr=" << r.threshold << ")" << (r.note.empty() ? "" : ("  " + r.note)) << "\n";
            }

            ASSERT_TRUE(pass) << "Correctness failed for " << e.name;
        }
    }
}

TEST(AllOps, Performance) {
    auto &reg = op_registry();
    ASSERT_FALSE(reg.empty()) << "No ops registered. Did you forget REGISTER_OP?";

    // 读取文件中的算子名
    std::unordered_set<std::string> operators_to_test = get_tested_operators("./test_op_name.sh");

    for (auto &e : reg) {
        if (operators_to_test.empty() || operators_to_test.find(e.name) != operators_to_test.end()) {
            std::cout << "\n========== PERF: " << e.name << " ==========\n";
            auto p = e.performance();

            std::cout << "\t" << "GPU time: " << p.ms << " ms\n";
            std::cout << "\t" << "CPU time: " << p.cpu_ms << " ms\n";
            if (!p.unit_name.empty()) {
                std::cout << "\t" << p.unit_name << ": " << p.unit_value << "\n";
            }
            std::cout << "\t" << "Input size: " << p.input_size / (1024 * 1024) << " MB\n";
            std::cout << "\t"
                      << "Output size: " << p.output_size / (1024 * 1024) << " MB\n";
            std::cout << "\tInput shape: ";
            for (const auto &shape : p.input_format) {
                std::cout << "{";
                for (size_t i = 0; i < shape.size(); ++i) {
                    std::cout << shape[i] << (i == shape.size() - 1 ? "" : ", ");
                }
                std::cout << "} ";
            }
            std::cout << "\n";
            std::cout << "\tOutput shape: ";
            for (const auto &shape : p.output_format) {
                std::cout << "{";
                for (size_t i = 0; i < shape.size(); ++i) {
                    std::cout << shape[i] << (i == shape.size() - 1 ? "" : ", ");
                }
                std::cout << "} ";
            }
            std::cout << "\n";
            if (!p.note.empty()) {
                std::cout << "\t" << "Note: " << p.note << "\n";
            }
        }
    }
}
