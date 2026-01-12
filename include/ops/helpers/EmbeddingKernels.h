#pragma once
#include "core/Tensor.h"

namespace OwnTensor {
namespace cuda {

void embedding_forward_cuda(
    const uint16_t* indices,
    const float* weight,
    float* output,
    int64_t N,
    int64_t C,
    int64_t V,
    int padding_idx
);

void embedding_backward_cuda(
    const uint16_t* indices,
    const float* grad_output,
    float* grad_weight,
    int64_t N,
    int64_t C,
    int64_t V,
    int padding_idx
);

} // namespace cuda
} // namespace OwnTensor                                                        