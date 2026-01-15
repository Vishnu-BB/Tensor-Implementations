// ============================================================================
// OPTIMIZED MATRIX MULTIPLICATION - CUDA-L2 Inspired Implementation
// ============================================================================
// Based on CUDA-L2 paper (arXiv:2512.02551) techniques:
// - Multi-level tiling (Block → Warp → Register)
// - Tensor Core acceleration with WMMA
// - Double buffering and pipelining
// - Advanced memory optimization
// Target: Surpass cuBLAS performance for FP16/BF16
// ============================================================================

#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <algorithm>

#include "ops/Matmul.cuh"
#include "core/Tensor.h"
#include "core/TensorDispatch.h"

namespace OwnTensor {

using namespace nvcuda;

// ============================================================================
// KERNEL CONFIGURATION - Optimized for A100/H100
// ============================================================================

// WMMA tile dimensions (hardware fixed)
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Block tile dimensions (tuned for Tensor Cores)
constexpr int BM = 128;  // Block tile M dimension
constexpr int BN = 128;  // Block tile N dimension
constexpr int BK = 32;   // Block tile K dimension

// Warp configuration
constexpr int WARP_SIZE = 32;
constexpr int WARPS_M = 4;  // 4 warps in M direction
constexpr int WARPS_N = 4;  // 4 warps in N direction
constexpr int NUM_WARPS = WARPS_M * WARPS_N;  // 16 warps per block
constexpr int THREADS_PER_BLOCK = NUM_WARPS * WARP_SIZE;  // 512 threads

// Warp tile dimensions
constexpr int WM = BM / WARPS_M;  // 32
constexpr int WN = BN / WARPS_N;  // 32

// Shared memory padding to avoid bank conflicts
constexpr int PAD = 8;

// For FP32/FP64 kernels
constexpr int BM_FP32 = 128;
constexpr int BN_FP32 = 128;
constexpr int BK_FP32 = 8;
constexpr int TM = 8;  // Thread tile M
constexpr int TN = 8;  // Thread tile N

// ============================================================================
// HELPER DEVICE FUNCTIONS
// ============================================================================

__device__ void compute_batch_offset(
    int batch_idx,
    const int* shape, const int* strides, int ndim,
    const int* out_shape, int out_ndim,
    int& offset)
{
    offset = 0;
    if (out_ndim <= 2) return;
    
    int temp_batch = batch_idx;
    for (int dim = out_ndim - 3; dim >= 0; --dim) {
        int batch_dim_size = out_shape[dim];
        int batch_coord = temp_batch % batch_dim_size;
        temp_batch /= batch_dim_size;
        
        int corres_dim = dim - (out_ndim - ndim);
        if (corres_dim >= 0 && corres_dim < ndim - 2) {
            int dim_size = shape[corres_dim];
            int idx = (dim_size > 1) ? batch_coord : 0;
            offset += idx * strides[corres_dim];
        }
    }
}

// ============================================================================
// FP16 TENSOR CORE KERNEL - CUDA-L2 Optimized
// ============================================================================

template<int BM, int BN, int BK, int WM, int WN>
__global__ void matmul_fp16_optimized(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K,
    int total_batches,
    const int* a_shape, const int* b_shape, const int* out_shape,
    const int* a_strides, const int* b_strides, const int* out_strides,
    int a_ndim, int b_ndim, int out_ndim)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int batch_idx = blockIdx.z;
    
    if (batch_idx >= total_batches) return;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int warp_row = warp_id / WARPS_N;
    const int warp_col = warp_id % WARPS_N;
    
    // Calculate batch offsets
    int a_batch_offset = 0, b_batch_offset = 0, out_batch_offset = 0;
    compute_batch_offset(batch_idx, a_shape, a_strides, a_ndim, out_shape, out_ndim, a_batch_offset);
    compute_batch_offset(batch_idx, b_shape, b_strides, b_ndim, out_shape, out_ndim, b_batch_offset);
    compute_batch_offset(batch_idx, out_shape, out_strides, out_ndim, out_shape, out_ndim, out_batch_offset);
    
    A += a_batch_offset;
    B += b_batch_offset;
    C += out_batch_offset;
    
    // Double-buffered shared memory with padding
    __shared__ __half As[2][BM][BK + PAD];
    __shared__ __half Bs[2][BK][BN + PAD];
    
    // WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> acc_frag[2][2];
    
    // Initialize accumulators
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::fill_fragment(acc_frag[i][j], __float2half(0.0f));
        }
    }
    
    const int block_row = by * BM;
    const int block_col = bx * BN;
    const int a_tile_stride = BM * BK;
    const int b_tile_stride = BK * BN;
    const int num_tiles = (K + BK - 1) / BK;
    
    int write_idx = 0;
    int read_idx = 1;
    
    // Prefetch first tile
    {
        for (int i = tid; i < a_tile_stride; i += THREADS_PER_BLOCK) {
            int row = i / BK;
            int col = i % BK;
            int global_row = block_row + row;
            int global_col = col;
            
            if (global_row < M && global_col < K) {
                As[write_idx][row][col] = A[global_row * K + global_col];
            } else {
                As[write_idx][row][col] = __float2half(0.0f);
            }
        }
        
        for (int i = tid; i < b_tile_stride; i += THREADS_PER_BLOCK) {
            int row = i / BN;
            int col = i % BN;
            int global_row = row;
            int global_col = block_col + col;
            
            if (global_row < K && global_col < N) {
                Bs[write_idx][row][col] = B[global_row * N + global_col];
            } else {
                Bs[write_idx][row][col] = __float2half(0.0f);
            }
        }
    }
    
    __syncthreads();
    
    // Main computation loop with double buffering
    for (int tile_k = 0; tile_k < num_tiles; tile_k++) {
        read_idx = write_idx;
        write_idx = 1 - write_idx;
        
        // Prefetch next tile
        if (tile_k + 1 < num_tiles) {
            int next_k = (tile_k + 1) * BK;
            
            for (int i = tid; i < a_tile_stride; i += THREADS_PER_BLOCK) {
                int row = i / BK;
                int col = i % BK;
                int global_row = block_row + row;
                int global_col = next_k + col;
                
                if (global_row < M && global_col < K) {
                    As[write_idx][row][col] = A[global_row * K + global_col];
                } else {
                    As[write_idx][row][col] = __float2half(0.0f);
                }
            }
            
            for (int i = tid; i < b_tile_stride; i += THREADS_PER_BLOCK) {
                int row = i / BN;
                int col = i % BN;
                int global_row = next_k + row;
                int global_col = block_col + col;
                
                if (global_row < K && global_col < N) {
                    Bs[write_idx][row][col] = B[global_row * N + global_col];
                } else {
                    Bs[write_idx][row][col] = __float2half(0.0f);
                }
            }
        }
        
        // Tensor Core computation
        #pragma unroll
        for (int k_step = 0; k_step < BK; k_step += WMMA_K) {
            #pragma unroll
            for (int wm = 0; wm < 2; wm++) {
                #pragma unroll
                for (int wn = 0; wn < 2; wn++) {
                    int warp_m_base = warp_row * WM + wm * WMMA_M;
                    int warp_n_base = warp_col * WN + wn * WMMA_N;
                    
                    wmma::load_matrix_sync(a_frag, &As[read_idx][warp_m_base][k_step], BK + PAD);
                    wmma::load_matrix_sync(b_frag, &Bs[read_idx][k_step][warp_n_base], BN + PAD);
                    wmma::mma_sync(acc_frag[wm][wn], a_frag, b_frag, acc_frag[wm][wn]);
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results to global memory
    #pragma unroll
    for (int wm = 0; wm < 2; wm++) {
        #pragma unroll
        for (int wn = 0; wn < 2; wn++) {
            int c_row = block_row + warp_row * WM + wm * WMMA_M;
            int c_col = block_col + warp_col * WN + wn * WMMA_N;
            
            if (c_row + WMMA_M <= M && c_col + WMMA_N <= N) {
                wmma::store_matrix_sync(&C[c_row * N + c_col], acc_frag[wm][wn], N, wmma::mem_row_major);
            } else if (c_row < M && c_col < N) {
                __half temp[WMMA_M * WMMA_N];
                wmma::store_matrix_sync(temp, acc_frag[wm][wn], WMMA_N, wmma::mem_row_major);
                
                for (int i = 0; i < WMMA_M; i++) {
                    for (int j = 0; j < WMMA_N; j++) {
                        if (c_row + i < M && c_col + j < N) {
                            C[(c_row + i) * N + (c_col + j)] = temp[i * WMMA_N + j];
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// BFLOAT16 TENSOR CORE KERNEL
// ============================================================================

template<int BM, int BN, int BK, int WM, int WN>
__global__ void matmul_bf16_optimized(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K,
    int total_batches,
    const int* a_shape, const int* b_shape, const int* out_shape,
    const int* a_strides, const int* b_strides, const int* out_strides,
    int a_ndim, int b_ndim, int out_ndim)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int batch_idx = blockIdx.z;
    
    if (batch_idx >= total_batches) return;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int warp_row = warp_id / WARPS_N;
    const int warp_col = warp_id % WARPS_N;
    
    // Calculate batch offsets
    int a_batch_offset = 0, b_batch_offset = 0, out_batch_offset = 0;
    compute_batch_offset(batch_idx, a_shape, a_strides, a_ndim, out_shape, out_ndim, a_batch_offset);
    compute_batch_offset(batch_idx, b_shape, b_strides, b_ndim, out_shape, out_ndim, b_batch_offset);
    compute_batch_offset(batch_idx, out_shape, out_strides, out_ndim, out_shape, out_ndim, out_batch_offset);
    
    A += a_batch_offset;
    B += b_batch_offset;
    C += out_batch_offset;
    
    __shared__ __nv_bfloat16 As[2][BM][BK + PAD];
    __shared__ __nv_bfloat16 Bs[2][BK][BN + PAD];
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag[2][2];
    
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::fill_fragment(acc_frag[i][j], 0.0f);
        }
    }
    
    const int block_row = by * BM;
    const int block_col = bx * BN;
    const int a_tile_stride = BM * BK;
    const int b_tile_stride = BK * BN;
    const int num_tiles = (K + BK - 1) / BK;
    
    int write_idx = 0;
    int read_idx = 1;
    
    // Prefetch first tile
    {
        for (int i = tid; i < a_tile_stride; i += THREADS_PER_BLOCK) {
            int row = i / BK;
            int col = i % BK;
            int global_row = block_row + row;
            int global_col = col;
            
            if (global_row < M && global_col < K) {
                As[write_idx][row][col] = A[global_row * K + global_col];
            } else {
                As[write_idx][row][col] = __float2bfloat16(0.0f);
            }
        }
        
        for (int i = tid; i < b_tile_stride; i += THREADS_PER_BLOCK) {
            int row = i / BN;
            int col = i % BN;
            int global_row = row;
            int global_col = block_col + col;
            
            if (global_row < K && global_col < N) {
                Bs[write_idx][row][col] = B[global_row * N + global_col];
            } else {
                Bs[write_idx][row][col] = __float2bfloat16(0.0f);
            }
        }
    }
    
    __syncthreads();
    
    // Main computation loop
    for (int tile_k = 0; tile_k < num_tiles; tile_k++) {
        read_idx = write_idx;
        write_idx = 1 - write_idx;
        
        // Prefetch next tile
        if (tile_k + 1 < num_tiles) {
            int next_k = (tile_k + 1) * BK;
            
            for (int i = tid; i < a_tile_stride; i += THREADS_PER_BLOCK) {
                int row = i / BK;
                int col = i % BK;
                int global_row = block_row + row;
                int global_col = next_k + col;
                
                if (global_row < M && global_col < K) {
                    As[write_idx][row][col] = A[global_row * K + global_col];
                } else {
                    As[write_idx][row][col] = __float2bfloat16(0.0f);
                }
            }
            
            for (int i = tid; i < b_tile_stride; i += THREADS_PER_BLOCK) {
                int row = i / BN;
                int col = i % BN;
                int global_row = next_k + row;
                int global_col = block_col + col;
                
                if (global_row < K && global_col < N) {
                    Bs[write_idx][row][col] = B[global_row * N + global_col];
                } else {
                    Bs[write_idx][row][col] = __float2bfloat16(0.0f);
                }
            }
        }
        
        // Tensor Core computation
        #pragma unroll
        for (int k_step = 0; k_step < BK; k_step += WMMA_K) {
            #pragma unroll
            for (int wm = 0; wm < 2; wm++) {
                #pragma unroll
                for (int wn = 0; wn < 2; wn++) {
                    int warp_m_base = warp_row * WM + wm * WMMA_M;
                    int warp_n_base = warp_col * WN + wn * WMMA_N;
                    
                    wmma::load_matrix_sync(a_frag, &As[read_idx][warp_m_base][k_step], BK + PAD);
                    wmma::load_matrix_sync(b_frag, &Bs[read_idx][k_step][warp_n_base], BN + PAD);
                    wmma::mma_sync(acc_frag[wm][wn], a_frag, b_frag, acc_frag[wm][wn]);
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results with FP32→BF16 conversion
    #pragma unroll
    for (int wm = 0; wm < 2; wm++) {
        #pragma unroll
        for (int wn = 0; wn < 2; wn++) {
            int c_row = block_row + warp_row * WM + wm * WMMA_M;
            int c_col = block_col + warp_col * WN + wn * WMMA_N;
            
            if (c_row < M && c_col < N) {
                float temp[WMMA_M * WMMA_N];
                wmma::store_matrix_sync(temp, acc_frag[wm][wn], WMMA_N, wmma::mem_row_major);
                
                for (int i = 0; i < WMMA_M; i++) {
                    for (int j = 0; j < WMMA_N; j++) {
                        if (c_row + i < M && c_col + j < N) {
                            C[(c_row + i) * N + (c_col + j)] = __float2bfloat16(temp[i * WMMA_N + j]);
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// FP32 2D BLOCKTILING KERNEL - Optimized for Register Tiling
// ============================================================================

template<int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_fp32_optimized(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int total_batches,
    const int* a_shape, const int* b_shape, const int* out_shape,
    const int* a_strides, const int* b_strides, const int* out_strides,
    int a_ndim, int b_ndim, int out_ndim)
{
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;
    const int batch_idx = blockIdx.z;
    
    if (batch_idx >= total_batches) return;
    
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);
    
    // Calculate batch offsets
    int a_batch_offset = 0, b_batch_offset = 0, out_batch_offset = 0;
    compute_batch_offset(batch_idx, a_shape, a_strides, a_ndim, out_shape, out_ndim, a_batch_offset);
    compute_batch_offset(batch_idx, b_shape, b_strides, b_ndim, out_shape, out_ndim, b_batch_offset);
    compute_batch_offset(batch_idx, out_shape, out_strides, out_ndim, out_shape, out_ndim, out_batch_offset);
    
    A += a_batch_offset;
    B += b_batch_offset;
    C += out_batch_offset;
    
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    
    float threadResults[TM * TN] = {0.0f};
    float regM[TM];
    float regN[TN];
    
    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Load A tile with vectorization
        for (int loadOffset = 0; loadOffset < BM; loadOffset += blockDim.x / (BK / 4)) {
            int innerRow = threadIdx.x / (BK / 4);
            int innerCol = (threadIdx.x % (BK / 4)) * 4;
            
            int aRow = cRow * BM + innerRow + loadOffset;
            int aCol = bkIdx + innerCol;
            
            if (aRow < M && aCol + 3 < K) {
                float4 tmp = *reinterpret_cast<const float4*>(&A[aRow * K + aCol]);
                As[(innerRow + loadOffset) * BK + innerCol + 0] = tmp.x;
                As[(innerRow + loadOffset) * BK + innerCol + 1] = tmp.y;
                As[(innerRow + loadOffset) * BK + innerCol + 2] = tmp.z;
                As[(innerRow + loadOffset) * BK + innerCol + 3] = tmp.w;
            } else if (aRow < M) {
                for (int i = 0; i < 4 && (aCol + i) < K; i++) {
                    As[(innerRow + loadOffset) * BK + innerCol + i] = A[aRow * K + aCol + i];
                }
            }
        }
        
        // Load B tile with vectorization
        for (int loadOffset = 0; loadOffset < BK; loadOffset += blockDim.x / (BN / 4)) {
            int innerRow = threadIdx.x / (BN / 4);
            int innerCol = (threadIdx.x % (BN / 4)) * 4;
            
            int bRow = bkIdx + innerRow + loadOffset;
            int bCol = cCol * BN + innerCol;
            
            if (bRow < K && bCol + 3 < N) {
                float4 tmp = *reinterpret_cast<const float4*>(&B[bRow * N + bCol]);
                Bs[(innerRow + loadOffset) * BN + innerCol + 0] = tmp.x;
                Bs[(innerRow + loadOffset) * BN + innerCol + 1] = tmp.y;
                Bs[(innerRow + loadOffset) * BN + innerCol + 2] = tmp.z;
                Bs[(innerRow + loadOffset) * BN + innerCol + 3] = tmp.w;
            } else if (bRow < K) {
                for (int i = 0; i < 4 && (bCol + i) < N; i++) {
                    Bs[(innerRow + loadOffset) * BN + innerCol + i] = B[bRow * N + bCol + i];
                }
            }
        }
        
        __syncthreads();
        
        // Compute
        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }
            
            #pragma unroll
            for (int i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }
            
            #pragma unroll
            for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                #pragma unroll
                for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results
    #pragma unroll
    for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
        #pragma unroll
        for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
            int outRow = cRow * BM + threadRow * TM + resIdxM;
            int outCol = cCol * BN + threadCol * TN + resIdxN;
            
            if (outRow < M && outCol < N) {
                C[outRow * N + outCol] = threadResults[resIdxM * TN + resIdxN];
            }
        }
    }
}

// ============================================================================
// FP64 2D BLOCKTILING KERNEL
// ============================================================================

template<int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_fp64_optimized(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    int M, int N, int K,
    int total_batches,
    const int* a_shape, const int* b_shape, const int* out_shape,
    const int* a_strides, const int* b_strides, const int* out_strides,
    int a_ndim, int b_ndim, int out_ndim)
{
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;
    const int batch_idx = blockIdx.z;
    
    if (batch_idx >= total_batches) return;
    
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);
    
    // Calculate batch offsets
    int a_batch_offset = 0, b_batch_offset = 0, out_batch_offset = 0;
    compute_batch_offset(batch_idx, a_shape, a_strides, a_ndim, out_shape, out_ndim, a_batch_offset);
    compute_batch_offset(batch_idx, b_shape, b_strides, b_ndim, out_shape, out_ndim, b_batch_offset);
    compute_batch_offset(batch_idx, out_shape, out_strides, out_ndim, out_shape, out_ndim, out_batch_offset);
    
    A += a_batch_offset;
    B += b_batch_offset;
    C += out_batch_offset;
    
    __shared__ double As[BM * BK];
    __shared__ double Bs[BK * BN];
    
    double threadResults[TM * TN] = {0.0};
    double regM[TM];
    double regN[TN];
    
    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Load A tile with double2 vectorization
        for (int loadOffset = 0; loadOffset < BM; loadOffset += blockDim.x / (BK / 2)) {
            int innerRow = threadIdx.x / (BK / 2);
            int innerCol = (threadIdx.x % (BK / 2)) * 2;
            
            int aRow = cRow * BM + innerRow + loadOffset;
            int aCol = bkIdx + innerCol;
            
            if (aRow < M && aCol + 1 < K) {
                double2 tmp = *reinterpret_cast<const double2*>(&A[aRow * K + aCol]);
                As[(innerRow + loadOffset) * BK + innerCol + 0] = tmp.x;
                As[(innerRow + loadOffset) * BK + innerCol + 1] = tmp.y;
            } else if (aRow < M && aCol < K) {
                As[(innerRow + loadOffset) * BK + innerCol] = A[aRow * K + aCol];
            }
        }
        
        // Load B tile with double2 vectorization
        for (int loadOffset = 0; loadOffset < BK; loadOffset += blockDim.x / (BN / 2)) {
            int innerRow = threadIdx.x / (BN / 2);
            int innerCol = (threadIdx.x % (BN / 2)) * 2;
            
            int bRow = bkIdx + innerRow + loadOffset;
            int bCol = cCol * BN + innerCol;
            
            if (bRow < K && bCol + 1 < N) {
                double2 tmp = *reinterpret_cast<const double2*>(&B[bRow * N + bCol]);
                Bs[(innerRow + loadOffset) * BN + innerCol + 0] = tmp.x;
                Bs[(innerRow + loadOffset) * BN + innerCol + 1] = tmp.y;
            } else if (bRow < K && bCol < N) {
                Bs[(innerRow + loadOffset) * BN + innerCol] = B[bRow * N + bCol];
            }
        }
        
        __syncthreads();
        
        // Compute
        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }
            
            #pragma unroll
            for (int i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }
            
            #pragma unroll
            for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                #pragma unroll
                for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results
    #pragma unroll
    for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
        #pragma unroll
        for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
            int outRow = cRow * BM + threadRow * TM + resIdxM;
            int outCol = cCol * BN + threadCol * TN + resIdxN;
            
            if (outRow < M && outCol < N) {
                C[outRow * N + outCol] = threadResults[resIdxM * TN + resIdxN];
            }
        }
    }
}

// ============================================================================
// HOST DISPATCH FUNCTIONS
// ============================================================================

template<typename T>
void launch_optimized_matmul(
    const Tensor& A, const Tensor& B, Tensor& output, cudaStream_t stream)
{
    const auto& a_shape = A.shape().dims;
    const auto& b_shape = B.shape().dims;
    const auto& out_shape = output.shape().dims;
    
    int a_ndim = static_cast<int>(a_shape.size());
    int b_ndim = static_cast<int>(b_shape.size());
    int out_ndim = static_cast<int>(out_shape.size());
    
    int M = static_cast<int>(a_shape[a_ndim - 2]);
    int K = static_cast<int>(a_shape[a_ndim - 1]);
    int N = static_cast<int>(b_shape[b_ndim - 1]);
    
    int total_batches = 1;
    for (int i = 0; i < out_ndim - 2; ++i) {
        total_batches *= static_cast<int>(out_shape[i]);
    }
    
    // Allocate device memory for metadata
    int *d_a_shape, *d_b_shape, *d_out_shape;
    int *d_a_strides, *d_b_strides, *d_out_strides;
    
    cudaMalloc(&d_a_shape, a_ndim * sizeof(int));
    cudaMalloc(&d_b_shape, b_ndim * sizeof(int));
    cudaMalloc(&d_out_shape, out_ndim * sizeof(int));
    cudaMalloc(&d_a_strides, a_ndim * sizeof(int));
    cudaMalloc(&d_b_strides, b_ndim * sizeof(int));
    cudaMalloc(&d_out_strides, out_ndim * sizeof(int));
    
    std::vector<int> a_shape_int(a_shape.begin(), a_shape.end());
    std::vector<int> b_shape_int(b_shape.begin(), b_shape.end());
    std::vector<int> out_shape_int(out_shape.begin(), out_shape.end());
    std::vector<int> a_strides_int(A.stride().strides.begin(), A.stride().strides.end());
    std::vector<int> b_strides_int(B.stride().strides.begin(), B.stride().strides.end());
    std::vector<int> out_strides_int(output.stride().strides.begin(), output.stride().strides.end());
    
    cudaMemcpyAsync(d_a_shape, a_shape_int.data(), a_ndim * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b_shape, b_shape_int.data(), b_ndim * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_out_shape, out_shape_int.data(), out_ndim * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_a_strides, a_strides_int.data(), a_ndim * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b_strides, b_strides_int.data(), b_ndim * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_out_strides, out_strides_int.data(), out_ndim * sizeof(int), cudaMemcpyHostToDevice, stream);
    
    const T* a_ptr = A.data<T>();
    const T* b_ptr = B.data<T>();
    T* out_ptr = output.data<T>();
    
    // Launch appropriate kernel based on type
    if constexpr (std::is_same<T, __half>::value) {
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, total_batches);
        
        matmul_fp16_optimized<BM, BN, BK, WM, WN><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(a_ptr),
            reinterpret_cast<const __half*>(b_ptr),
            reinterpret_cast<__half*>(out_ptr),
            M, N, K, total_batches,
            d_a_shape, d_b_shape, d_out_shape,
            d_a_strides, d_b_strides, d_out_strides,
            a_ndim, b_ndim, out_ndim
        );
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, total_batches);
        
        matmul_bf16_optimized<BM, BN, BK, WM, WN><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(a_ptr),
            reinterpret_cast<const __nv_bfloat16*>(b_ptr),
            reinterpret_cast<__nv_bfloat16*>(out_ptr),
            M, N, K, total_batches,
            d_a_shape, d_b_shape, d_out_shape,
            d_a_strides, d_b_strides, d_out_strides,
            a_ndim, b_ndim, out_ndim
        );
    } else if constexpr (std::is_same<T, float>::value) {
        dim3 block(256);  // (BM/TM) * (BN/TN) = 16 * 16 = 256
        dim3 grid((N + BN_FP32 - 1) / BN_FP32, (M + BM_FP32 - 1) / BM_FP32, total_batches);
        
        matmul_fp32_optimized<BM_FP32, BN_FP32, BK_FP32, TM, TN><<<grid, block, 0, stream>>>(
            a_ptr, b_ptr, out_ptr,
            M, N, K, total_batches,
            d_a_shape, d_b_shape, d_out_shape,
            d_a_strides, d_b_strides, d_out_strides,
            a_ndim, b_ndim, out_ndim
        );
    } else if constexpr (std::is_same<T, double>::value) {
        dim3 block(256);
        dim3 grid((N + BN_FP32 - 1) / BN_FP32, (M + BM_FP32 - 1) / BM_FP32, total_batches);
        
        matmul_fp64_optimized<BM_FP32, BN_FP32, BK_FP32, TM, TN><<<grid, block, 0, stream>>>(
            a_ptr, b_ptr, out_ptr,
            M, N, K, total_batches,
            d_a_shape, d_b_shape, d_out_shape,
            d_a_strides, d_b_strides, d_out_strides,
            a_ndim, b_ndim, out_ndim
        );
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_a_shape); cudaFree(d_b_shape); cudaFree(d_out_shape);
        cudaFree(d_a_strides); cudaFree(d_b_strides); cudaFree(d_out_strides);
        throw std::runtime_error("Optimized matmul kernel failed: " + std::string(cudaGetErrorString(err)));
    }
    
    // Free device memory
    cudaFree(d_a_shape);
    cudaFree(d_b_shape);
    cudaFree(d_out_shape);
    cudaFree(d_a_strides);
    cudaFree(d_b_strides);
    cudaFree(d_out_strides);
}

// ============================================================================
// UNIFIED PUBLIC API - cuda_matmul_optimized
// ============================================================================

void cuda_matmul(const Tensor& A, const Tensor& B, Tensor& output, cudaStream_t stream)
{
// CRITICAL: Make inputs contiguous - .t() creates non-contiguous views
    // The kernel at line 496 (A) and 517 (B) assumes row-major layout
    // Without this, transposed tensors read garbage/inf values
    Tensor A_contig = A.is_contiguous() ? A : A.contiguous();
    Tensor B_contig = B.is_contiguous() ? B : B.contiguous();
    
    dispatch_by_dtype(A.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        
        if constexpr (std::is_same<T, __half>::value || 
                      std::is_same<T, __nv_bfloat16>::value ||
                      std::is_same<T, float>::value ||
                      std::is_same<T, double>::value) {
            launch_optimized_matmul<T>(A_contig, B_contig, output, stream);
        } else {
            throw std::runtime_error("cuda_matmul_optimized: Unsupported data type. Supported: FP16, BF16, FP32, FP64");
        }
    });
    
    // Synchronize to ensure kernel completes before A_contig/B_contig go out of scope
    cudaStreamSynchronize(stream);
}

} // namespace OwnTensor

#endif // WITH_CUDA