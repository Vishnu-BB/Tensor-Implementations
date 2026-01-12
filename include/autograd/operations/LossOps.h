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

/**
 * @brief Autograd-aware sparse cross entropy with logits
 * 
 * Takes logits (B, ..., V) and targets (B, ...) where V is vocab size.
 * Returns the loss per element.
 */
Tensor sparse_cross_entropy_with_logits(const Tensor& logits, const Tensor& targets);

} // namespace autograd
} // namespace OwnTensor
