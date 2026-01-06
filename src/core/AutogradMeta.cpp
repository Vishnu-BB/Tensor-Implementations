#include "core/AutogradMeta.h"
#include "core/Tensor.h"
#include "core/TensorImpl.h"
#include <stdexcept>

namespace OwnTensor {

// ============================================================================
// Constructors
// ============================================================================

AutogradMeta::AutogradMeta(bool requires_grad)
    : requires_grad_(requires_grad) {
    // grad_ is nullptr - lazy allocation
}

AutogradMeta::AutogradMeta(AutogradMeta&& other) noexcept
    : grad_(std::move(other.grad_)),
      requires_grad_(other.requires_grad_) {
    // mutex is not movable, but that's fine - each AutogradMeta has its own
}

AutogradMeta& AutogradMeta::operator=(AutogradMeta&& other) noexcept {
    if (this != &other) {
        std::lock_guard<std::mutex> lock1(mutex_);
        std::lock_guard<std::mutex> lock2(other.mutex_);
        
        grad_ = std::move(other.grad_);
        requires_grad_ = other.requires_grad_;
    }
    return *this;
}

// ============================================================================
// Gradient Access
// ============================================================================

Tensor& AutogradMeta::mutable_grad(TensorImpl* self_impl) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!grad_) {
        if (!self_impl) {
            throw std::runtime_error("AutogradMeta::mutable_grad: self_impl is null");
        }
        
        // Lazy allocation: create gradient tensor with same shape/dtype/device
        grad_ = std::make_unique<Tensor>(
            self_impl->sizes(),
            self_impl->dtype(),
            self_impl->device(),
            false  // gradient itself doesn't require grad
        );
    }
    
    return *grad_;
}

const Tensor& AutogradMeta::grad() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!grad_) {
        throw std::runtime_error("AutogradMeta::grad: gradient has not been allocated");
    }
    
    return *grad_;
}

void AutogradMeta::set_grad(const Tensor& new_grad) {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_ = std::make_unique<Tensor>(new_grad);
}

bool AutogradMeta::has_grad() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return grad_ != nullptr;
}

void AutogradMeta::reset_grad() {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_.reset();
}

// ============================================================================
// Requires Grad
// ============================================================================

void AutogradMeta::set_requires_grad(bool requires_grad) {
    std::lock_guard<std::mutex> lock(mutex_);
    requires_grad_ = requires_grad;
}

} // namespace OwnTensor
