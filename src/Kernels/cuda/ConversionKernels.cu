// ============================================================================
// Create new file: src/core/ConversionKernels.cu
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include "dtype/Types.h"
#include "ops/helpers/ConversionKernels.cuh"
namespace OwnTensor {


// ============================================================================
// CUDA Kernel: Convert any type to bool
// ============================================================================

template<typename T>
__global__ void convert_to_bool_kernel(const T* __restrict__ input,
                                        bool* __restrict__ output,
                                        int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        output[idx] = (input[idx] != T(0.0f));
    }
}

// Specialization for __half (GPU native type)
template<>
__global__ void convert_to_bool_kernel<__half>(const __half* __restrict__ input,
                                                 bool* __restrict__ output,
                                                 int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = __half2float(input[idx]);
        output[idx] = (val != 0.0f);
    }
}

// Specialization for __nv_bfloat16 (GPU native type)
template<>
__global__ void convert_to_bool_kernel<__nv_bfloat16>(const __nv_bfloat16* __restrict__ input,
                                                        bool* __restrict__ output,
                                                        int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = __bfloat162float(input[idx]);
        output[idx] = (val != 0.0f);
    }
}

// ============================================================================
// Host function to launch kernel (with type conversion)
// ============================================================================
template<typename T>
void convert_to_bool_cuda(const T* input, bool* output, int64_t n, cudaStream_t stream) {
    if (n == 0) return;
    
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    // Type conversion for half types
    if constexpr (std::is_same_v<T, float16_t>) {
        const __half* gpu_input = reinterpret_cast<const __half*>(input);
        convert_to_bool_kernel<__half><<<blocks, threads, 0, stream>>>(gpu_input, output, n);
    } 
    else if constexpr (std::is_same_v<T, bfloat16_t>) {
        const __nv_bfloat16* gpu_input = reinterpret_cast<const __nv_bfloat16*>(input);
        convert_to_bool_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(gpu_input, output, n);
    }
    else {
        convert_to_bool_kernel<T><<<blocks, threads, 0, stream>>>(input, output, n);
    }
    
    //  Check launch errors
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("convert_to_bool_cuda kernel launch failed: ") + 
            cudaGetErrorString(launch_err)
        );
    }
    
    //  ADD: Synchronize to ensure completion
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    if (sync_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("convert_to_bool_cuda synchronization failed: ") + 
            cudaGetErrorString(sync_err)
        );
    }
}  

// ============================================================================
// Explicit template instantiations
// ============================================================================

template void convert_to_bool_cuda<bool>(const bool*, bool*, int64_t, cudaStream_t);
template void convert_to_bool_cuda<int16_t>(const int16_t*, bool*, int64_t, cudaStream_t);
template void convert_to_bool_cuda<int32_t>(const int32_t*, bool*, int64_t, cudaStream_t);
template void convert_to_bool_cuda<int64_t>(const int64_t*, bool*, int64_t, cudaStream_t);
template void convert_to_bool_cuda<float>(const float*, bool*, int64_t, cudaStream_t);
template void convert_to_bool_cuda<double>(const double*, bool*, int64_t, cudaStream_t);
template void convert_to_bool_cuda<float16_t>(const float16_t*, bool*, int64_t, cudaStream_t);
template void convert_to_bool_cuda<bfloat16_t>(const bfloat16_t*, bool*, int64_t, cudaStream_t);
template void convert_to_bool_cuda<uint8_t>(const uint8_t*, bool*, int64_t, cudaStream_t);
template void convert_to_bool_cuda<uint16_t>(const uint16_t*, bool*, int64_t, cudaStream_t);
template void convert_to_bool_cuda<uint32_t>(const uint32_t*, bool*, int64_t, cudaStream_t);
template void convert_to_bool_cuda<uint64_t>(const uint64_t*, bool*, int64_t, cudaStream_t);
template void convert_to_bool_cuda<complex32_t>(const complex32_t*, bool*, int64_t, cudaStream_t);
template void convert_to_bool_cuda<complex64_t>(const complex64_t*, bool*, int64_t, cudaStream_t);
template void convert_to_bool_cuda<complex128_t>(const complex128_t*, bool*, int64_t, cudaStream_t);

} // namespace OwnTensor

// ============================================================================
// Generic Type Conversion Kernel
// ============================================================================
namespace OwnTensor {

template<typename Dst>
__global__ void convert_type_kernel(const float* __restrict__ input,
                                  Dst* __restrict__ output,
                                  int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = static_cast<Dst>(input[idx]);
    }
}

// Specialization for float -> float16_t (using __half)
template<>
__global__ void convert_type_kernel<float16_t>(const float* __restrict__ input,
                                                    float16_t* __restrict__ output,
                                                    int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Use CUDA intrinsic for conversion
        __half h = __float2half(input[idx]);
        // Store as raw bits
        output[idx].raw_bits = *reinterpret_cast<uint16_t*>(&h);
    }
}

// Specialization for float -> bfloat16_t (using __nv_bfloat16)
template<>
__global__ void convert_type_kernel<bfloat16_t>(const float* __restrict__ input,
                                                     bfloat16_t* __restrict__ output,
                                                     int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Use CUDA intrinsic for conversion
        __nv_bfloat16 b = __float2bfloat16(input[idx]);
        // Store as raw bits
        output[idx].raw_bits = *reinterpret_cast<uint16_t*>(&b);
    }
}

template<typename Dst>
void convert_type_cuda(const float* input, Dst* output, int64_t n, cudaStream_t stream) {
    if (n == 0) return;
    
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    convert_type_kernel<Dst><<<blocks, threads, 0, stream>>>(input, output, n);
    
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("convert_type_cuda kernel launch failed: ") + 
            cudaGetErrorString(launch_err)
        );
    }
}

// Explicit instantiations
// Did not work without the instantiations
template void convert_type_cuda<float16_t>(const float*, float16_t*, int64_t, cudaStream_t);
template void convert_type_cuda<bfloat16_t>(const float*, bfloat16_t*, int64_t, cudaStream_t);

} // namespace OwnTensor

