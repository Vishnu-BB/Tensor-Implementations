#pragma once

#include "core/Tensor.h"
#include <cuda_runtime.h>

namespace OwnTensor {
namespace cuda {

/**
 * @brief Compute sparse cross entropy loss (forward pass).
 * 
 * loss = sum(-log(softmax(logits)[targets])) / batch_size
 * Uses numerically stable log-softmax computation.
 * 
 * @param logits [batch_size, vocab_size] - Input logits
 * @param targets [batch_size] - Target indices
 * @param loss_output [1] - Output scalar loss (sum of all sample losses)
 * @param batch_size Number of samples
 * @param vocab_size Number of classes
 * @param stream CUDA stream for asynchronous execution
 */
template<typename T, typename T_idx>
void sparse_cross_entropy_forward_cuda(
    const T* logits,
    const T_idx* targets,
    T* loss_output,
    int64_t batch_size,
    int64_t vocab_size,
    cudaStream_t stream
);

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
