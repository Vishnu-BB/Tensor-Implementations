#include "autograd/operations/LossOps.h"
#include "autograd/ops_template.h"
#include "autograd/backward/LossBackward.h"
#include "ops/TensorOps.h"
#include "ops/UnaryOps/Reduction.h"
#include "ops/helpers/ConditionalOps.h"
#include "ops/UnaryOps/Exponents.h"
#include "ops/UnaryOps/Arithmetics.h"

namespace OwnTensor {
namespace autograd {

Tensor mse_loss(const Tensor& predictions, const Tensor& targets) {
    // Forward: mean((pred - target)^2)
    Tensor diff = predictions - targets;
    Tensor sq_diff = OwnTensor::pow(diff, 2, 0);
    Tensor result = reduce_mean(sq_diff);
    
    // Build graph if predictions require grad
    if (predictions.requires_grad()) {
        auto grad_fn = std::make_shared<MSELossBackward>(predictions, targets, predictions.numel());
        Tensor& pred_mut = const_cast<Tensor&>(predictions);
        grad_fn->set_next_edge(0, get_grad_edge(pred_mut));
        result.set_grad_fn(grad_fn);
        result.set_requires_grad(true);
    }
    
    return result;
}

Tensor mae_loss(const Tensor& predictions, const Tensor& targets) {
    // Forward: mean(|pred - target|)
    Tensor diff = predictions - targets;
    Tensor abs_diff = OwnTensor::abs(diff, 0);
    Tensor result = reduce_mean(abs_diff);
    
    // Build graph if predictions require grad
    if (predictions.requires_grad()) {
        auto grad_fn = std::make_shared<MAELossBackward>(predictions, targets, predictions.numel());
        Tensor& pred_mut = const_cast<Tensor&>(predictions);
        grad_fn->set_next_edge(0, get_grad_edge(pred_mut));
        result.set_grad_fn(grad_fn);
        result.set_requires_grad(true);
    }
    
    return result;
}

Tensor binary_cross_entropy(const Tensor& predictions, const Tensor& targets) {
    float epsilon_val = 1e-7f;
    Tensor epsilon = Tensor::full(predictions.shape(), 
        TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()), epsilon_val);
    Tensor one_minus_epsilon = Tensor::full(predictions.shape(), 
        TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()), 1.0f - epsilon_val);

    // Clip predictions
    Tensor clipped_preds = where((predictions < epsilon), epsilon, predictions);
    clipped_preds = where((clipped_preds > one_minus_epsilon), one_minus_epsilon, clipped_preds);

    Tensor term1 = targets * OwnTensor::log(clipped_preds);
    Tensor ones = Tensor::ones(targets.shape(), 
        TensorOptions().with_device(targets.device()).with_dtype(targets.dtype()));
    Tensor term2 = (ones - targets) * OwnTensor::log(ones - clipped_preds);
    Tensor sum_terms = term1 + term2;
    Tensor neg_one = Tensor::full({{1}}, 
        TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()), -1.0f);
    Tensor result = reduce_mean(sum_terms) * neg_one;
    
    // Build graph if predictions require grad
    if (predictions.requires_grad()) {
        auto grad_fn = std::make_shared<BCELossBackward>(clipped_preds, targets, predictions.numel());
        Tensor& pred_mut = const_cast<Tensor&>(predictions);
        grad_fn->set_next_edge(0, get_grad_edge(pred_mut));
        result.set_grad_fn(grad_fn);
        result.set_requires_grad(true);
    }
    
    return result;
}

Tensor categorical_cross_entropy(const Tensor& predictions, const Tensor& targets) {
    float epsilon_val = 1e-7f;
    Tensor epsilon = Tensor::full(predictions.shape(), 
        TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()), epsilon_val);
    Tensor one_minus_epsilon = Tensor::full(predictions.shape(), 
        TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()), 1.0f - epsilon_val);

    // Clip predictions
    Tensor clipped_preds = where((predictions < epsilon), epsilon, predictions);
    clipped_preds = where((clipped_preds > one_minus_epsilon), one_minus_epsilon, clipped_preds);

    Tensor log_preds = OwnTensor::log(clipped_preds);
    Tensor target_log_probs = targets * log_preds;
    
    std::vector<int64_t> axis = {1};
    Tensor sample_losses = reduce_sum(target_log_probs, axis);
    Tensor neg_one = Tensor::full({{1}}, 
        TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()), -1.0f);
    Tensor result = reduce_mean(sample_losses) * neg_one;
    
    // Build graph if predictions require grad
    if (predictions.requires_grad()) {
        auto grad_fn = std::make_shared<CCELossBackward>(clipped_preds, targets, predictions.numel());
        Tensor& pred_mut = const_cast<Tensor&>(predictions);
        grad_fn->set_next_edge(0, get_grad_edge(pred_mut));
        result.set_grad_fn(grad_fn);
        result.set_requires_grad(true);
    }
    
    return result;
}

} // namespace autograd
} // namespace OwnTensor
