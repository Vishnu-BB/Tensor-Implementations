// ============================================================================
// Create new file: include/core/ConversionKernels.cuh
// ============================================================================

#pragma once

// #ifdef WITH_CUDA
#include <cuda_runtime.h>
#include <cstdint>

namespace OwnTensor {


// Forward declarations
template<typename T>
void convert_to_bool_cuda(const T* input, bool* output, int64_t n, cudaStream_t stream = 0);

template<typename Dst>
void convert_type_cuda(const float* input, Dst* output, int64_t n, cudaStream_t stream = 0);


// #endif // WITH_CUDA
} // namespace OwnTensor
