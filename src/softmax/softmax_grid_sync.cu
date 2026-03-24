#include <algorithm>
#include <cfloat>
#include <cooperative_groups.h>

#include "op_runtime.hpp"

// 辅助函数
__device__ __forceinline__ void get_max_sum(float& sum_self, float& max_self){
    #pragma unroll
    for(int offset = 16; offset > 0; offset /= 2){
        float sum_other = __shfl_down_sync(0xffffffff, sum_self, offset);
        float max_other = __shfl_down_sync(0xffffffff, max_self, offset);
        float max_temp = fmaxf(max_other, max_self);
        
        // Online Softmax 修正公式
        sum_self = sum_self * __expf(max_self - max_temp) + sum_other * __expf(max_other - max_temp);
        max_self = max_temp;
    }
}

// 存储所有线程块的max和sum
__device__ float g_block_max[1024]; // 足够容纳所有 SM 的输出
__device__ float g_block_sum[1024];

// 所有线程块内部进行online softmax 得到局部sum和max
template<int threadsPerBlock>
__global__ void softmax_grid_sync_kernel(const float* input, float* output, int N) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    
    // 共享内存：用来存放每个 Warp 规约后的结果
    __shared__ float smem_data[(threadsPerBlock / 32) * 2];
    
    int tid = threadIdx.x;
    int id = blockDim.x * blockIdx.x + tid;
    int stride = gridDim.x * blockDim.x;

    float max_self = -FLT_MAX;
    float sum_self = 0.0f;

    // 网格跨步循环：每个线程扫一遍自己负责的所有数据
    for (int i = id; i < N; i += stride) {
        float val = input[i];
        float max_prev = max_self;
        max_self = fmaxf(max_prev, val);
        sum_self = sum_self * __expf(max_prev - max_self) + __expf(val - max_self);
    }

    // warp内部规约
    get_max_sum(sum_self, max_self);

    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = threadsPerBlock / 32;

    if (lane_id == 0) {
        smem_data[warp_id] = sum_self;
        smem_data[warp_id + num_warps] = max_self;
    }
    __syncthreads();

    // 二次规约求出block的局部sum和max
    if (warp_id == 0) {
        float b_sum = (tid < num_warps) ? smem_data[tid] : 0.0f;
        float b_max = (tid < num_warps) ? smem_data[tid + num_warps] : -FLT_MAX;
        get_max_sum(b_sum, b_max);
        
        // 数据写入到全局内存
        if (tid == 0) {
            g_block_max[blockIdx.x] = b_max;
            g_block_sum[blockIdx.x] = b_sum;
        }
    }
    
    grid.sync();

    __shared__ float final_G_max, final_G_sum;

    // 由第一个 Warp 负责汇总那 blocksPerGrid 个块的局部情报
    if (threadIdx.x < 32) {
        float b_max = -FLT_MAX;
        float b_sum = 0.0f;
        
        // 获取所有线程块的数据
        for(int i = threadIdx.x; i < gridDim.x; i += 32) {
            float m = g_block_max[i];
            float s = g_block_sum[i];
            float m_prev = b_max;
            b_max = fmaxf(m_prev, m);
            b_sum = b_sum * __expf(m_prev - b_max) + s * __expf(m - b_max);
        }
        
        // Warp 内再次规约对齐
        get_max_sum(b_sum, b_max);
        if (threadIdx.x == 0) {
            final_G_max = b_max;
            final_G_sum = b_sum;
        }
    }
    __syncthreads(); // 此时块内所有线程都拿到了全局真值

    // 输出结果
    id = blockDim.x * blockIdx.x + threadIdx.x;
    stride = gridDim.x * blockDim.x;
    for (int i = id; i < N; i += stride) {
        // 利用全局 Max 和 Sum 归一化
        output[i] = __expf(input[i] - final_G_max) / final_G_sum;
    }
}


extern "C" void solve(const float* input, float* output, int N) {
    const int threadsPerBlock = 512;

    // 获取设备信息
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    int sm_count = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));

    int supportsCoopLaunch = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, device));
    if (!supportsCoopLaunch) {
        std::cerr << "Cooperative launch is not supported on this device.\n";
        std::exit(EXIT_FAILURE);
    }

    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm,
        softmax_grid_sync_kernel<threadsPerBlock>,
        threadsPerBlock,
        0
    ));
    if (blocks_per_sm <= 0) {
        std::cerr << "Failed to derive cooperative occupancy for softmax_grid_sync.\n";
        std::exit(EXIT_FAILURE);
    }

    // Cooperative launch 的 block 数不能超过设备能同时驻留的总 block 数。
    int blocksPerGrid = std::min((N + threadsPerBlock - 1) / threadsPerBlock, sm_count * blocks_per_sm);

    // 准备 kernel 参数数组
    void* args[] = { (void*)&input, (void*)&output, (void*)&N };

    // 协作式内核启动
    dim3 gridDim(blocksPerGrid, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);
    CUDA_CHECK(cudaLaunchCooperativeKernel(
        (void*)softmax_grid_sync_kernel<threadsPerBlock>,
        gridDim,
        blockDim,
        args,
        0,       // shared memory
        0        // stream 0
    ));

    CUDA_CHECK(cudaGetLastError());
}

LC_REGISTER_SOFTMAX_1D_OP("softmax_grid_sync", opfw::cpu_softmax_1d_f32, solve);
