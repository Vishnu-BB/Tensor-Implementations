#include "autograd/Functions.h"
#include "ops/TensorOps.h"
#include "ops/ScalarOps.h"
#include "ops/Kernels.h"
#include "core/AutogradMeta.h"
#include <stdexcept>

namespace OwnTensor {
namespace autograd {

// ============================================================================
// AddBackward
// ============================================================================

std::vector<Tensor> AddBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("AddBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    // grad_a = grad_output
    // grad_b = grad_output
    // Both inputs get the same gradient
    return {grad_output, grad_output};
}

// ============================================================================
// MulBackward
// ============================================================================

MulBackward::MulBackward(const Tensor& a, const Tensor& b)
    : Node(2), saved_a_(a), saved_b_(b) {}

std::vector<Tensor> MulBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("MulBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    // grad_a = grad_output * b
    // grad_b = grad_output * a
    Tensor grad_a = grad_output * saved_b_;
    Tensor grad_b = grad_output * saved_a_;
    
    return {grad_a, grad_b};
}

// ============================================================================
// MatmulBackward
// ============================================================================

MatmulBackward::MatmulBackward(const Tensor& a, const Tensor& b)
    : Node(2), saved_a_(a), saved_b_(b) {}

std::vector<Tensor> MatmulBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("MatmulBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    // grad_a = grad_output @ b.T
    // grad_b = a.T @ grad_output
    Tensor b_t = saved_b_.t();
    Tensor a_t = saved_a_.t();
    
    Tensor grad_a = matmul(grad_output, b_t);
    Tensor grad_b = matmul(a_t, grad_output);
    
    return {grad_a, grad_b};
}

// ============================================================================
// ReluBackward
// ============================================================================

ReluBackward::ReluBackward(const Tensor& input)
    : Node(1), saved_input_(input) {}

std::vector<Tensor> ReluBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("ReluBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    // grad_input = grad_output * (input > 0)
    Tensor mask = saved_input_ > 0.0f;
    Tensor grad_input = grad_output * mask;
    
    return {grad_input};
}

// ============================================================================
// SumBackward
// ============================================================================

SumBackward::SumBackward(const Shape& input_shape)
    : Node(1), input_shape_(input_shape) {}

std::vector<Tensor> SumBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("SumBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    // Broadcast grad_output to input_shape
    // For sum, gradient is just the grad_output expanded to original shape
    // TODO: Implement proper broadcasting/expand
    // For now, assume grad_output is scalar and input_shape is larger
    
    Tensor grad_input = Tensor::ones(input_shape_, 
        TensorOptions()
            .with_dtype(grad_output.dtype())
            .with_device(grad_output.device()));
    
    // Scale by grad_output value (if scalar)
    if (grad_output.ndim() == 0 || grad_output.numel() == 1) {
        const float* grad_val = grad_output.data<float>();
        grad_input = grad_input * (*grad_val);
    }
    
    return {grad_input};
}

// ============================================================================
// MeanBackward
// ============================================================================

MeanBackward::MeanBackward(const Shape& input_shape, int64_t numel)
    : Node(1), input_shape_(input_shape), numel_(numel) {}

std::vector<Tensor> MeanBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("MeanBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    // grad_input = grad_output / numel
    Tensor grad_input = Tensor::ones(input_shape_,
        TensorOptions()
            .with_dtype(grad_output.dtype())
            .with_device(grad_output.device()));
    
    // Scale by grad_output / numel
    if (grad_output.ndim() == 0 || grad_output.numel() == 1) {
        const float* grad_val = grad_output.data<float>();
        float scale = (*grad_val) / static_cast<float>(numel_);
        grad_input = grad_input * scale;
    }
    
    return {grad_input};
}

// ============================================================================
// GradAccumulator
// ============================================================================

GradAccumulator::GradAccumulator(TensorImpl* impl)
    : Node(1), leaf_impl_(impl) {}

std::vector<Tensor> GradAccumulator::apply(std::vector<Tensor>&& grads) {
    if (grads.empty() || !leaf_impl_) {
        return {};
    }
    
    const Tensor& grad_output = grads[0];
    
    // Accumulate gradient into leaf tensor
    if (leaf_impl_->has_autograd_meta()) {
        auto* meta = static_cast<AutogradMeta*>(leaf_impl_->autograd_meta());
        
        if (meta->has_grad()) {
            // Accumulate: existing_grad += grad_output
            Tensor& existing_grad = meta->mutable_grad(leaf_impl_);
            Tensor new_grad = operator+(existing_grad, grad_output);
            meta->set_grad(new_grad);
        } else {
            // First gradient: just set it
            meta->set_grad(grad_output);
        }
    }
    
    // No outputs (leaf node)
    return {};
}

} // namespace autograd
} // namespace OwnTensor
