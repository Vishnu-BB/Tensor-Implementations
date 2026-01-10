#pragma once

#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Autograd-aware MSE loss
 */
Tensor mse_loss(const Tensor& predictions, const Tensor& targets);

/**
 * @brief Autograd-aware MAE loss
 */
Tensor mae_loss(const Tensor& predictions, const Tensor& targets);

/**
 * @brief Autograd-aware binary cross entropy loss
 */
Tensor binary_cross_entropy(const Tensor& predictions, const Tensor& targets);

/**
 * @brief Autograd-aware categorical cross entropy loss
 */
Tensor categorical_cross_entropy(const Tensor& predictions, const Tensor& targets);

} // namespace autograd
} // namespace OwnTensor
