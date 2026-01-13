#include "nn/optimizer/Optim.h"
#include "ops/TensorOps.h"
#include "ops/ScalarOps.h"
#include "ops/UnaryOps/Arithmetics.h"
#include "ops/UnaryOps/Reduction.h"
#include <cmath>
#include <iostream>
namespace OwnTensor {
namespace nn {

Optimizer::Optimizer(const std::vector<Tensor>& params) : params_(params) {
    for (const auto& p : params_) {
        void* key = p.unsafeGetTensorImpl();
        if (p.requires_grad() && p.dtype() != Dtype::Float32) {
            master_params_[key] = p.as_type(Dtype::Float32);
        }
    }
}

void Optimizer::zero_grad() {
    for (auto& p : params_) {
        if (p.requires_grad()) {
            p.fill_grad(0.0f);
        }
    }
}

Tensor* Optimizer::get_master_weight(const Tensor& v) {
    void* key = v.unsafeGetTensorImpl();
    auto it = master_params_.find(key);
    if (it != master_params_.end()) {
        return &it->second;
    }
    return nullptr;
}

SGDOptimizer::SGDOptimizer(const std::vector<Tensor>& params, float learning_rate)
    : Optimizer(params), learning_rate_(learning_rate) {}

void SGDOptimizer::step() {
    for (auto& p : params_) {
        if (!p.requires_grad()) continue;

        void* key = p.unsafeGetTensorImpl();
        Tensor grad_f32 = p.grad_view();
        if (grad_f32.dtype() != Dtype::Float32) {
            grad_f32 = grad_f32.as_type(Dtype::Float32);
        }

        auto it_master = master_params_.find(key);
        if (it_master != master_params_.end()) {
            Tensor& master_p = it_master->second;
            master_p += -learning_rate_ * grad_f32;
            p.copy_(master_p.as_type(p.dtype()));
        } else {
            // p is already Float32 or we update it directly
            // Note: If p is not Float32 and no master param, we might lose precision.
            // But the constructor ensures master_params for non-Float32.
            p += -learning_rate_ * grad_f32;
        }
    }
}

// **************************************************************************************
// ======================== Adam Optimizer ==============================================
// **************************************************************************************

Adam::Adam(std::vector<Tensor*> params, 
           float lr, 
           float beta1, 
           float beta2,
           float eps,
           float weight_decay)
    : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), 
      weight_decay_(weight_decay), step_count_(0),
      params_(std::move(params)), initialized_(false) {}

void Adam::step() {
    step_count_++;
    
    // Lazy initialization of momentum buffers
    if (!initialized_) {
        m_.reserve(params_.size());
        v_.reserve(params_.size());
        for (auto* param : params_) {
            TensorOptions opts = TensorOptions()
                .with_dtype(param->dtype())
                .with_device(param->device());
            m_.push_back(Tensor::zeros(param->shape(), opts));
            v_.push_back(Tensor::zeros(param->shape(), opts));
        }
        initialized_ = true;
    }
    
    // Bias correction factors
    float bias_correction1 = 1.0f - std::pow(beta1_, static_cast<float>(step_count_));
    float bias_correction2 = 1.0f - std::pow(beta2_, static_cast<float>(step_count_));
    
    for (size_t i = 0; i < params_.size(); ++i) {
        Tensor* param = params_[i];
        if (!param->requires_grad() || !param->has_grad()) {
            continue;
        }
        Tensor grad;
        try{
            grad=param->grad_view();
        } catch(...){
            continue;
        }
        int64_t numel = param->numel();
        
        // CPU implementation with direct data manipulation
        if (param->device().is_cpu()) {
            float* param_data = param->data<float>();
            const float* grad_data = param->grad<float>();
            float* m_data = m_[i].data<float>();
            float* v_data = v_[i].data<float>();
            
            for (int64_t j = 0; j < numel; ++j) {
                float g = grad_data[j];
                
                // Apply weight decay (AdamW style)
                if (weight_decay_ > 0.0f) {
                    g += weight_decay_ * param_data[j];
                }
                
                // Update first moment: m = beta1 * m + (1 - beta1) * g
                m_data[j] = beta1_ * m_data[j] + (1.0f - beta1_) * g;
                
                // Update second moment: v = beta2 * v + (1 - beta2) * g^2
                v_data[j] = beta2_ * v_data[j] + (1.0f - beta2_) * g * g;
                
                // Bias-corrected estimates
                float m_hat = m_data[j] / bias_correction1;
                float v_hat = v_data[j] / bias_correction2;
                
                // Update parameter
                param_data[j] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
            }
        } else {
            // For CUDA: transfer to CPU, update, transfer back
            Tensor param_cpu = param->to_cpu();
            Tensor grad_cpu = grad.to_cpu();
            Tensor m_cpu = m_[i].to_cpu();
            Tensor v_cpu = v_[i].to_cpu();
            
            float* param_data = param_cpu.data<float>();
            const float* grad_data = grad_cpu.data<float>();
            float* m_data = m_cpu.data<float>();
            float* v_data = v_cpu.data<float>();
            
            for (int64_t j = 0; j < numel; ++j) {
                float g = grad_data[j];
                
                if (weight_decay_ > 0.0f) {
                    g += weight_decay_ * param_data[j];
                }
                
                m_data[j] = beta1_ * m_data[j] + (1.0f - beta1_) * g;
                v_data[j] = beta2_ * v_data[j] + (1.0f - beta2_) * g * g;
                
                float m_hat = m_data[j] / bias_correction1;
                float v_hat = v_data[j] / bias_correction2;
                
                param_data[j] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
            }
            
            // Copy back
            param->copy_(param_cpu);
            m_[i].copy_(m_cpu);
            v_[i].copy_(v_cpu);
        }
    }
}

void Adam::zero_grad() {
    for (auto* param : params_) {
        // Try to zero gradient - use try-catch for CUDA compatibility
        try {
            param->zero_grad();
        } catch (...) {
            // No gradient to zero
        }
    }
}

float clip_grad_norm_(std::vector<Tensor*>& params, float max_norm) {
    // Compute global gradient norm
    float total_norm_sq = 0.0f;
    
    for (auto* param : params) {
        if (!param->requires_grad()) {
            continue;
        }
        
        // Try to get gradient - use try-catch since grad_view might throw
        Tensor grad;
        try {
            grad = param->grad_view();
        } catch (...) {
            continue;  // No gradient for this parameter
        }
        
        // Make sure gradient is contiguous before access
        Tensor grad_cont = grad.is_contiguous() ? grad : grad.contiguous();
        
        // Transfer to CPU for computation
        Tensor grad_cpu = grad_cont.device().is_cpu() ? grad_cont : grad_cont.to_cpu();
        const float* grad_data = grad_cpu.data<float>();
        int64_t numel = grad_cpu.numel();
        
        float param_norm_sq = 0.0f;
        for (int64_t i = 0; i < numel; ++i) {
            param_norm_sq += grad_data[i] * grad_data[i];
        }
        total_norm_sq += param_norm_sq;
    }
    
    float total_norm = std::sqrt(total_norm_sq);
    
    // Clip if necessary
    if (total_norm > max_norm) {
        float clip_coef = max_norm / (total_norm + 1e-6f);
        
        for (auto* param : params) {
            if (!param->requires_grad()) {
                continue;
            }
            
            // Try to get gradient
            Tensor grad;
            try {
                grad = param->grad_view();  
            } catch (...) {
                continue;
            }
            
            // Scale gradient in-place
            if (param->device().is_cpu()) {
                float* grad_data = param->grad<float>();
                int64_t numel = param->numel();
                for (int64_t i = 0; i < numel; ++i) {
                    grad_data[i] *= clip_coef;
                }
            } else {
                // For CUDA: get grad view, scale on CPU, copy back
                Tensor grad_cpu = grad.to_cpu();
                float* grad_data = grad_cpu.data<float>();
                int64_t numel = grad_cpu.numel();
                for (int64_t i = 0; i < numel; ++i) {
                    grad_data[i] *= clip_coef;
                }
                // Copy scaled gradient back in-place
                grad.copy_(grad_cpu);
            }
        }
    }
    
    return total_norm;
}

// *********************************************************************************************
// ============================ AdamW Optimizer ================================================
// *********************************************************************************************

AdamW::AdamW(const std::vector<Tensor>& params, float alpha, float beta1, float beta2, float epsilon, float weight_decay)
   : Optimizer(params), alpha_(alpha), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), weight_decay_(weight_decay), t_(0) {
  
   for (const auto& p : params_) {
       if (p.requires_grad()) {
           void* key = p.unsafeGetTensorImpl();
           TensorOptions opts_f32 = TensorOptions().with_dtype(Dtype::Float32).with_device(p.device());
           m_[key] = Tensor::zeros(p.shape(), opts_f32);
           v_[key] = Tensor::zeros(p.shape(), opts_f32);
       }
   }
}

void AdamW::step() {
   t_++;
   float bias_corr1 = 1.0f - std::pow(beta1_, t_);
   float bias_corr2 = 1.0f - std::pow(beta2_, t_);
   
   for (auto& p : params_) {
       if (!p.requires_grad()) continue;

       void* key = p.unsafeGetTensorImpl();
       Tensor grad_f32 = p.grad_view();
       if (grad_f32.dtype() != Dtype::Float32) {
           grad_f32 = grad_f32.as_type(Dtype::Float32);
       }

       Tensor& m = m_[key];
       Tensor& v = v_[key];

       m *= beta1_;
       m += (1.0f - beta1_) * grad_f32;
       v *= beta2_;
       v += (1.0f - beta2_) * OwnTensor::square(grad_f32);

       Tensor p_f32;
       auto it_master = master_params_.find(key);
       bool is_mixed = (it_master != master_params_.end());
      
       if (is_mixed) {
           p_f32 = it_master->second;
       } else {
           p_f32 = p;
           if (p_f32.dtype() != Dtype::Float32) {
               p_f32 = p_f32.as_type(Dtype::Float32);
           }
       }

       float alpha_eff = alpha_ * std::sqrt(bias_corr2) / bias_corr1;
       Tensor update = (alpha_eff * m / (OwnTensor::sqrt(v) + epsilon_ * std::sqrt(bias_corr2)))
                       + (alpha_ * weight_decay_ * p_f32);

       if (is_mixed) {
           Tensor& master_p = it_master->second;
           master_p -= update;
           p.copy_(master_p.as_type(p.dtype()));
       } else {
           p -= update;
       }
   }
}

} // namespace nn
} // namespace OwnTensor
