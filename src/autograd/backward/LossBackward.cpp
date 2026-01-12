#include "autograd/backward/LossBackward.h"
#include "autograd/operations/ActivationOps.h"
#include "ops/TensorOps.h"
#include "ops/ScalarOps.h"
#include "ops/helpers/ConditionalOps.h"
#include "ops/helpers/LossKernels.h"
#include "device/DeviceCore.h"
#include <stdexcept>
#include <cmath>
#include <iostream>

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

// ============================================================================
// SparseCrossEntropyBackward
// ============================================================================

SparseCrossEntropyBackward::SparseCrossEntropyBackward(const Tensor& logits, const Tensor& target, int64_t dim)
    : Node(1), saved_logits_(logits), saved_target_(target), dim_(dim) {}

std::vector<Tensor> SparseCrossEntropyBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("SparseCrossEntropyBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    // grad_logits = (softmax(logits) - one_hot(target)) * grad_output
    Tensor probs = softmax(saved_logits_, dim_);
    
    int64_t num_samples = saved_target_.numel();
    float scale = 1.0f / static_cast<float>(num_samples);
    
    // Get scalar grad_output value
    float grad_val = 1.0f;
    if (grad_output.numel() == 1) {
        if (grad_output.is_cuda()) {
            grad_val = grad_output.to_cpu().data<float>()[0];
        } else {
            grad_val = *grad_output.data<float>();
        }
    }
    
    scale *= grad_val;
    
    if (saved_target_.is_cpu()) {
        Tensor probs = softmax(saved_logits_, dim_);
        Tensor grad_logits = probs * scale;
        
        dispatch_by_dtype(grad_logits.dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            T* g_ptr = grad_logits.data<T>();
            
            dispatch_by_integer_dtype(saved_target_.dtype(), [&](auto dummy_idx) {
                using T_idx = decltype(dummy_idx);
                const T_idx* t_ptr = saved_target_.data<T_idx>();
                
                int64_t batch_size = saved_target_.numel();
                int64_t vocab_size = saved_logits_.shape().dims.back();
                
                #pragma omp parallel for
                for (int64_t i = 0; i < batch_size; ++i) {
                    int64_t target_idx = static_cast<int64_t>(t_ptr[i]);
                    if (target_idx >= 0 && target_idx < vocab_size) {
                        g_ptr[i * vocab_size + target_idx] -= static_cast<T>(scale);
                    }
                }
            });
        });
        return {grad_logits};
    } else {
#ifdef WITH_CUDA
        Tensor grad_logits = Tensor::zeros(saved_logits_.shape(), 
            TensorOptions().with_device(saved_logits_.device()).with_dtype(saved_logits_.dtype()));
            
        cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
        int64_t batch_size = saved_target_.numel();
        int64_t vocab_size = saved_logits_.shape().dims.back();

        dispatch_by_dtype(saved_logits_.dtype(), [&](auto dummy) {
            using T = decltype(dummy);
            if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
                dispatch_by_integer_dtype(saved_target_.dtype(), [&](auto dummy_idx) {
                    using T_idx = decltype(dummy_idx);
                    cuda::sparse_cross_entropy_backward_cuda<T, T_idx>(
                        saved_logits_.data<T>(),
                        saved_target_.data<T_idx>(),
                        grad_logits.data<T>(),
                        batch_size,
                        vocab_size,
                        static_cast<T>(scale),
                        stream
                    );
                });
            } else {
                throw std::runtime_error("SparseCrossEntropyBackward: unsupported logits type");
            }
        });
        return {grad_logits};
#else
        throw std::runtime_error("SparseCrossEntropyBackward: CUDA implementation not yet provided");
#endif
    }
}

} // namespace autograd
} // namespace OwnTensor
