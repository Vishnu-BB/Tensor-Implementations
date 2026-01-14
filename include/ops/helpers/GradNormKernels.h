#pragma once
#include <cstdint>

namespace OwnTensor {
namespace cuda {

/**
 * @brief Compute sum of squared elements for a gradient tensor
 * Adds result to an accumulator (atomically)
 */
void grad_norm_squared_cuda(
    const float* grad,
    float* norm_sq_accumulator,  // Single float on GPU to accumulate into
    int64_t numel
);

/**
 * @brief Scale all gradients by a coefficient
 */
void scale_gradients_cuda(
    float* grad,
    float scale,
    int64_t numel
);

} // namespace cuda
} // namespace OwnTensor
