#include "autograd/backward/LossBackward.h"
#include "ops/TensorOps.h"
#include "ops/ScalarOps.h"
#include "ops/helpers/ConditionalOps.h"
#include <stdexcept>

namespace OwnTensor {
namespace autograd {

// ============================================================================
// MSELossBackward
// ============================================================================

MSELossBackward::MSELossBackward(const Tensor& pred, const Tensor& target, int64_t numel)
    : Node(1), saved_pred_(pred), saved_target_(target), numel_(numel) {}

std::vector<Tensor> MSELossBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("MSELossBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    // grad_pred = 2 * (pred - target) / numel * grad_output
    Tensor diff = saved_pred_ - saved_target_;
    float scale = 2.0f / static_cast<float>(numel_);
    
    // Get scalar grad_output value
    float grad_val = 1.0f;
    if (grad_output.numel() == 1) {
        if (grad_output.is_cuda()) {
            grad_val = grad_output.to_cpu().data<float>()[0];
        } else {
            grad_val = *grad_output.data<float>();
        }
    }
    
    Tensor grad_pred = diff * (scale * grad_val);
    
    return {grad_pred};
}

// ============================================================================
// MAELossBackward
// ============================================================================

MAELossBackward::MAELossBackward(const Tensor& pred, const Tensor& target, int64_t numel)
    : Node(1), saved_pred_(pred), saved_target_(target), numel_(numel) {}

std::vector<Tensor> MAELossBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("MAELossBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    // grad_pred = sign(pred - target) / numel * grad_output
    Tensor diff = saved_pred_ - saved_target_;
    Tensor zero = Tensor::zeros(diff.shape(), 
        TensorOptions().with_dtype(diff.dtype()).with_device(diff.device()));
    Tensor ones = Tensor::ones(diff.shape(),
        TensorOptions().with_dtype(diff.dtype()).with_device(diff.device()));
    Tensor neg_ones = ones * -1.0f;
    
    // sign(x) = 1 if x > 0, -1 if x < 0, 0 if x == 0
    Tensor sign_diff = where(diff > zero, ones, where(diff < zero, neg_ones, zero));
    
    float scale = 1.0f / static_cast<float>(numel_);
    
    // Get scalar grad_output value
    float grad_val = 1.0f;
    if (grad_output.numel() == 1) {
        if (grad_output.is_cuda()) {
            grad_val = grad_output.to_cpu().data<float>()[0];
        } else {
            grad_val = *grad_output.data<float>();
        }
    }
    
    Tensor grad_pred = sign_diff * (scale * grad_val);
    
    return {grad_pred};
}

// ============================================================================
// BCELossBackward
// ============================================================================

BCELossBackward::BCELossBackward(const Tensor& pred, const Tensor& target, int64_t numel)
    : Node(1), saved_pred_(pred), saved_target_(target), numel_(numel) {}

std::vector<Tensor> BCELossBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("BCELossBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    // BCE: L = -mean(target * log(pred) + (1-target) * log(1-pred))
    // grad_pred = (-target/pred + (1-target)/(1-pred)) / numel
    Tensor ones = Tensor::ones(saved_pred_.shape(),
        TensorOptions().with_dtype(saved_pred_.dtype()).with_device(saved_pred_.device()));
    
    Tensor term1 = saved_target_ / saved_pred_ * -1.0f;
    Tensor term2 = (ones - saved_target_) / (ones - saved_pred_);
    
    float scale = 1.0f / static_cast<float>(numel_);
    
    // Get scalar grad_output value
    float grad_val = 1.0f;
    if (grad_output.numel() == 1) {
        if (grad_output.is_cuda()) {
            grad_val = grad_output.to_cpu().data<float>()[0];
        } else {
            grad_val = *grad_output.data<float>();
        }
    }
    
    Tensor grad_pred = (term1 + term2) * (scale * grad_val);
    
    return {grad_pred};
}

// ============================================================================
// CCELossBackward
// ============================================================================

CCELossBackward::CCELossBackward(const Tensor& pred, const Tensor& target, int64_t numel)
    : Node(1), saved_pred_(pred), saved_target_(target), numel_(numel) {}

std::vector<Tensor> CCELossBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("CCELossBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    // CCE: L = -mean(sum(target * log(pred), dim=1))
    // grad_pred = -target / pred / numel
    Tensor grad_pred = saved_target_ / saved_pred_ * -1.0f;
    
    float scale = 1.0f / static_cast<float>(numel_);
    
    // Get scalar grad_output value
    float grad_val = 1.0f;
    if (grad_output.numel() == 1) {
        if (grad_output.is_cuda()) {
            grad_val = grad_output.to_cpu().data<float>()[0];
        } else {
            grad_val = *grad_output.data<float>();
        }
    }
    
    grad_pred = grad_pred * (scale * grad_val);
    
    return {grad_pred};
}

} // namespace autograd
} // namespace OwnTensor