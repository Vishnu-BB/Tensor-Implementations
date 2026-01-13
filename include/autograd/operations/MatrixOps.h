#pragma once

#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Autograd-aware matrix multiplication
 */
Tensor matmul(const Tensor& a, const Tensor& b);



} // namespace autograd
} // namespace OwnTensor