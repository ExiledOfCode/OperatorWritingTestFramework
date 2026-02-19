// gpu_props.cu
// 用途：打印当前机器上 CUDA GPU 的关键硬件属性（写 kernel 常用参考数据）
// 编译：nvcc -O2 gpu_props.cu -o gpu_props
// 运行：./gpu_props

#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <string>

// 简单的错误检查宏
#define CUDA_CHECK(call)                                                                                                                                       \
    do {                                                                                                                                                       \
        cudaError_t err = (call);                                                                                                                              \
        if (err != cudaSuccess) {                                                                                                                              \
            std::cerr << "CUDA 错误: " << cudaGetErrorString(err) << " @ " << __FILE__ << ":" << __LINE__ << std::endl;                                        \
            std::exit(1);                                                                                                                                      \
        }                                                                                                                                                      \
    } while (0)

// 把字节数转换成更易读的 KB/MB/GB 字符串
static std::string bytes_to_human(size_t bytes) {
    const double KB = 1024.0;
    const double MB = KB * 1024.0;
    const double GB = MB * 1024.0;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);

    if (bytes >= (size_t)GB) {
        oss << (bytes / GB) << " GB";
    } else if (bytes >= (size_t)MB) {
        oss << (bytes / MB) << " MB";
    } else if (bytes >= (size_t)KB) {
        oss << (bytes / KB) << " KB";
    } else {
        oss << bytes << " B";
    }
    return oss.str();
}

int main() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cout << "没有检测到 CUDA GPU。\n";
        return 0;
    }

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    std::cout << "================ GPU Kernel 编写参考信息 ================\n";
    std::cout << "当前使用设备 ID: " << dev << "\n";
    std::cout << "设备名称: " << prop.name << "\n";
    std::cout << "计算能力 (Compute Capability): " << prop.major << "." << prop.minor << "\n\n";

    // 1) SM/线程/warp 等基础信息
    std::cout << "---- 基础并行结构 ----\n";
    std::cout << "SM 数量 (multiProcessorCount): " << prop.multiProcessorCount << "\n";
    std::cout << "Warp 大小 (warpSize): " << prop.warpSize << "\n";
    std::cout << "每个 SM 最大线程数 (maxThreadsPerMultiProcessor): " << prop.maxThreadsPerMultiProcessor << "\n";
    std::cout << "每个 Block 最大线程数 (maxThreadsPerBlock): " << prop.maxThreadsPerBlock << "\n";
    std::cout << "每个 Block 的最大维度 (maxThreadsDim): " << prop.maxThreadsDim[0] << " x " << prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2] << "\n";
    std::cout << "Grid 最大维度 (maxGridSize): " << prop.maxGridSize[0] << " x " << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << "\n\n";

    // 2) 内存层级：shared、L2、global、constant 等
    std::cout << "---- 内存与缓存 ----\n";
    std::cout << "全局显存总量 (totalGlobalMem): " << bytes_to_human(prop.totalGlobalMem) << "\n";
    std::cout << "每个 SM 的 shared memory (sharedMemPerMultiprocessor): " << bytes_to_human(prop.sharedMemPerMultiprocessor) << "\n";
    std::cout << "每个 Block 的 shared memory 上限 (sharedMemPerBlock): " << bytes_to_human(prop.sharedMemPerBlock) << "\n";
    std::cout << "每个 Block 的可用寄存器数量 (regsPerBlock): " << prop.regsPerBlock << "\n";
    std::cout << "每个 SM 的可用寄存器数量 (regsPerMultiprocessor): " << prop.regsPerMultiprocessor << "\n";
    std::cout << "L2 缓存大小 (l2CacheSize): " << bytes_to_human(prop.l2CacheSize) << "\n";
    std::cout << "常量内存大小 (totalConstMem): " << bytes_to_human(prop.totalConstMem) << "\n";
    std::cout << "内存总线位宽 (memoryBusWidth): " << prop.memoryBusWidth << " bits\n";
    std::cout << "显存时钟 (memoryClockRate): " << prop.memoryClockRate / 1000.0 << " MHz\n";
    std::cout << "GPU 核心时钟 (clockRate): " << prop.clockRate / 1000.0 << " MHz\n\n";

    // 3) 访存相关：对齐、纹理、pitch、memcpy 等
    std::cout << "---- 访存/对齐/纹理等 ----\n";
    std::cout << "纹理对齐 (textureAlignment): " << prop.textureAlignment << " bytes\n";
    std::cout << "纹理 pitch 对齐 (texturePitchAlignment): " << prop.texturePitchAlignment << " bytes\n";
    std::cout << "最大 pitch (memPitch): " << bytes_to_human(prop.memPitch) << "\n";
    std::cout << "是否支持统一寻址 (unifiedAddressing): " << (prop.unifiedAddressing ? "是" : "否") << "\n";
    std::cout << "是否支持 Managed Memory (managedMemory): " << (prop.managedMemory ? "是" : "否") << "\n";
    std::cout << "是否支持 pageableMemoryAccess: " << (prop.pageableMemoryAccess ? "是" : "否") << "\n";
    std::cout << "是否支持 concurrentManagedAccess: " << (prop.concurrentManagedAccess ? "是" : "否") << "\n\n";

    // 4) 并发/流/拷贝能力
    std::cout << "---- 并发与拷贝能力 ----\n";
    std::cout << "是否支持异步拷贝与 kernel 并发 (deviceOverlap): " << (prop.deviceOverlap ? "是" : "否") << "\n";
    std::cout << "支持的异步引擎数 (asyncEngineCount): " << prop.asyncEngineCount << "\n";
    std::cout << "是否支持 kernel 并发 (concurrentKernels): " << (prop.concurrentKernels ? "是" : "否") << "\n";
    std::cout << "可用的最大流优先级范围 (streamPrioritiesSupported): " << (prop.streamPrioritiesSupported ? "支持" : "不支持") << "\n\n";

    // 5) 原子/指令/执行能力（部分字段可能取决于 CUDA 版本）
    std::cout << "---- 指令/原子等能力 ----\n";
    std::cout << "是否支持 native double 原子加 (atomicAdd on double): " << (prop.major >= 6 ? "通常支持(取决于架构/实现)" : "可能不支持/需注意") << "\n\n";

    // 6) 推荐的 occupancy 相关信息（只是硬件上限，实际还要看寄存器/shared
    // 使用）
    std::cout << "---- Occupancy/调度参考 ----\n";
    std::cout << "每个 SM 最大 block 数 (maxBlocksPerMultiProcessor): " << prop.maxBlocksPerMultiProcessor << "\n";
    std::cout << "每个 SM 最大 warp 数（推导值）: " << (prop.maxThreadsPerMultiProcessor / prop.warpSize) << "\n";
    std::cout << "注意：实际 occupancy 会受寄存器/共享内存/指令等影响。\n\n";

    // 7) 列出所有 GPU（可选但很实用）
    std::cout << "================ 本机所有 CUDA 设备概览 ================\n";
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp p{};
        CUDA_CHECK(cudaGetDeviceProperties(&p, i));
        std::cout << "[" << i << "] " << p.name << " | CC " << p.major << "." << p.minor << " | SM " << p.multiProcessorCount << " | GlobalMem "
                  << bytes_to_human(p.totalGlobalMem) << "\n";
    }

    std::cout << "========================================================\n";
    return 0;
}
