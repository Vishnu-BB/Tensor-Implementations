#pragma once

#include "core/Tensor.h"
#include <cuda_runtime.h>

namespace OwnTensor {
namespace cuda {

/**
 * @brief Compute gradient for Sparse Cross Entropy with Logits.
 * 
 * grad = (softmax(logits) - targets_one_hot) * scale
 */
template<typename T, typename T_idx>
void sparse_cross_entropy_backward_cuda(
    const T* logits,
    const T_idx* targets,
    T* grad_logits,
    int64_t batch_size,
    int64_t vocab_size,
    T scale,
    cudaStream_t stream
);

} // namespace cuda
} // namespace OwnTensor
