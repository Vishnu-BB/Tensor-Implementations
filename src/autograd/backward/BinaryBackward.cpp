#include "autograd/backward/BinaryBackward.h"
#include "ops/TensorOps.h"
#include "ops/ScalarOps.h"
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
    
    // grad_a = grad_output, grad_b = grad_output
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
    
    // grad_a = grad_output * b, grad_b = grad_output * a
    Tensor grad_a = grad_output * saved_b_;
    Tensor grad_b = grad_output * saved_a_;
    
    return {grad_a, grad_b};
}

// ============================================================================
// SubBackward
// ============================================================================

std::vector<Tensor> SubBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("SubBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    // grad_a = grad_output, grad_b = -grad_output
    Tensor neg_grad = grad_output * -1.0f;
    return {grad_output, neg_grad};
}

// ============================================================================
// DivBackward
// ============================================================================

DivBackward::DivBackward(const Tensor& a, const Tensor& b)
    : Node(2), saved_a_(a), saved_b_(b) {}

std::vector<Tensor> DivBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("DivBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    // grad_a = grad_output / b
    Tensor grad_a = grad_output / saved_b_;
    
    // grad_b = -grad_output * a / b^2
    Tensor term1 = grad_output * -1.0f;
    Tensor term2 = term1 * saved_a_;
    Tensor b_sq = saved_b_ * saved_b_;
    Tensor grad_b = term2 / b_sq;
    
    return {grad_a, grad_b};
}


} // namespace autograd
} // namespace OwnTensor
