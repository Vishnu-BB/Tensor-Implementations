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
        grad_val = *grad_output.data<float>();
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
        grad_val = *grad_output.data<float>();
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
        grad_val = *grad_output.data<float>();
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
        grad_val = *grad_output.data<float>();
    }
    
    grad_pred = grad_pred * (scale * grad_val);
    
    return {grad_pred};
}

// ============================================================================
// SparseCrossEntropyLossBackward
// ============================================================================

SparseCrossEntropyLossBackward::SparseCrossEntropyLossBackward(
    const Tensor& logits, const Tensor& targets, int64_t batch_size, int64_t num_classes)
    : Node(1), saved_logits_(logits), saved_targets_(targets), 
      batch_size_(batch_size), num_classes_(num_classes) {}

std::vector<Tensor> SparseCrossEntropyLossBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("SparseCrossEntropyLossBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    // Get scalar grad_output value - must transfer to CPU if on CUDA
    float grad_val = 1.0f;
    if (grad_output.numel() == 1) {
        Tensor grad_cpu = grad_output.device().is_cpu() ? grad_output : grad_output.to_cpu();
        grad_val = *grad_cpu.data<float>();
    }
    
    // Handle both 2D [N, C] and 3D [B, T, C] logits
    // We need to flatten to 2D for computation, then reshape back
    auto logits_shape = saved_logits_.shape().dims;
    Shape original_shape = saved_logits_.shape();
    
    Tensor logits_2d = saved_logits_;
    if (logits_shape.size() == 3) {
        // Flatten [B, T, C] -> [B*T, C]
        logits_2d = saved_logits_.view(Shape{{batch_size_, num_classes_}});
    }
    
    // Compute softmax and gradient on 2D tensor
    TensorOptions opts = TensorOptions()
        .with_dtype(saved_logits_.dtype())
        .with_device(saved_logits_.device());
    
    Tensor grad_logits_2d = Tensor::zeros(logits_2d.shape(), opts);
    
    // CPU implementation
    if (logits_2d.device().is_cpu()) {
        const float* logits_data = logits_2d.data<float>();
        float* grad_data = grad_logits_2d.data<float>();
        
        // Compute softmax and gradient for each sample
        for (int64_t i = 0; i < batch_size_; ++i) {
            // Find max for numerical stability
            float max_val = logits_data[i * num_classes_];
            for (int64_t c = 1; c < num_classes_; ++c) {
                max_val = std::max(max_val, logits_data[i * num_classes_ + c]);
            }
            
            // Compute exp(x - max) and sum
            float sum_exp = 0.0f;
            std::vector<float> exp_vals(num_classes_);
            for (int64_t c = 0; c < num_classes_; ++c) {
                exp_vals[c] = std::exp(logits_data[i * num_classes_ + c] - max_val);
                sum_exp += exp_vals[c];
            }
            
            // Get target class
            int64_t target_class = 0;
            if (saved_targets_.dtype() == Dtype::Int64) {
                target_class = saved_targets_.data<int64_t>()[i];
            } else if (saved_targets_.dtype() == Dtype::Int32) {
                target_class = static_cast<int64_t>(saved_targets_.data<int32_t>()[i]);
            } else if (saved_targets_.dtype() == Dtype::UInt16) {
                target_class = static_cast<int64_t>(saved_targets_.data<uint16_t>()[i]);
            }
            
            // Gradient: softmax[i,c] - (c == target ? 1 : 0)
            float scale = grad_val / static_cast<float>(batch_size_);
            for (int64_t c = 0; c < num_classes_; ++c) {
                float softmax_val = exp_vals[c] / sum_exp;
                float one_hot = (c == target_class) ? 1.0f : 0.0f;
                grad_data[i * num_classes_ + c] = (softmax_val - one_hot) * scale;
            }
        }
    } else {
        // CUDA: transfer to CPU, compute, transfer back
        Tensor logits_cpu = logits_2d.to_cpu();
        Tensor targets_cpu = saved_targets_.to_cpu();
        Tensor grad_cpu = Tensor::zeros(logits_2d.shape(), 
            TensorOptions().with_dtype(saved_logits_.dtype()));
        
        const float* logits_data = logits_cpu.data<float>();
        float* grad_data = grad_cpu.data<float>();
        
        for (int64_t i = 0; i < batch_size_; ++i) {
            float max_val = logits_data[i * num_classes_];
            for (int64_t c = 1; c < num_classes_; ++c) {
                max_val = std::max(max_val, logits_data[i * num_classes_ + c]);
            }
            
            float sum_exp = 0.0f;
            std::vector<float> exp_vals(num_classes_);
            for (int64_t c = 0; c < num_classes_; ++c) {
                exp_vals[c] = std::exp(logits_data[i * num_classes_ + c] - max_val);
                sum_exp += exp_vals[c];
            }
            
            int64_t target_class = 0;
            if (targets_cpu.dtype() == Dtype::Int64) {
                target_class = targets_cpu.data<int64_t>()[i];
            } else if (targets_cpu.dtype() == Dtype::UInt16) {
                target_class = static_cast<int64_t>(targets_cpu.data<uint16_t>()[i]);
            }
            
            float scale = grad_val / static_cast<float>(batch_size_);
            for (int64_t c = 0; c < num_classes_; ++c) {
                float softmax_val = exp_vals[c] / sum_exp;
                float one_hot = (c == target_class) ? 1.0f : 0.0f;
                grad_data[i * num_classes_ + c] = (softmax_val - one_hot) * scale;
            }
        }
        
        grad_logits_2d = grad_cpu.to(saved_logits_.device());
    }
    
    // Reshape gradient back to original shape if needed
    Tensor grad_logits = grad_logits_2d;
    if (logits_shape.size() == 3) {
        grad_logits = grad_logits_2d.view(original_shape);
    }
    
    return {grad_logits};
}

} // namespace autograd
} // namespace OwnTensor