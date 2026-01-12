#include "nn/optimizer/Optim.h"
#include "ops/TensorOps.h"
#include "ops/ScalarOps.h"
#include "ops/UnaryOps/Arithmetics.h"
#include "ops/UnaryOps/Reduction.h"
#include <cmath>

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

Adam::Adam(const std::vector<Tensor>& params, float alpha, float beta1, float beta2, float epsilon)
    : Optimizer(params), alpha_(alpha), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0) {
    
    for (const auto& p : params_) {
        if (p.requires_grad()) {
            void* key = p.unsafeGetTensorImpl();
            TensorOptions opts_f32 = TensorOptions().with_dtype(Dtype::Float32).with_device(p.device());
            
            m_[key] = Tensor::zeros(p.shape(), opts_f32);
            v_[key] = Tensor::zeros(p.shape(), opts_f32);
        }
    }
}

void Adam::step() {
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

        float alpha_eff = alpha_ * std::sqrt(bias_corr2) / bias_corr1;
        Tensor update = alpha_eff * m / (OwnTensor::sqrt(v) + epsilon_ * std::sqrt(bias_corr2));

        auto it_master = master_params_.find(key);
        if (it_master != master_params_.end()) {
            Tensor& master_p = it_master->second;
            master_p -= update;
            p.copy_(master_p.as_type(p.dtype()));
        } else {
            p -= update;
        }
    }
}

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
