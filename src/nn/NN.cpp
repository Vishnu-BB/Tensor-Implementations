#include "nn/NN.h"
#include "autograd/AutogradOps.h"
#include "ops/ScalarOps.h"  // For operator*(Tensor, float)
#include "ops/TensorOps.h"
#include <cmath>

namespace OwnTensor {
namespace nn {

// ============================================================================
// Module
// ============================================================================

std::vector<Tensor> Module::parameters() {
    return params_;
}

void Module::to(DeviceIndex dev) {
    // We override to() in leaf modules to update members.
    // Base class to() just iterates over parameters and moves them.
    // Note: To actually update members, we need to assign back.
    // Since parameters() returns by value, we need a better way.
    // Let's just make to() virtual as well if needed, or let 
    // leaf modules handle it.
    for (auto& p : parameters()) {
        p = p.to(dev);
    }
}

void Module::zero_grad() {
    for (auto& p : parameters()) {
        // Only attempt to zero gradients if they require grad and exist
        if (p.requires_grad()) {
            try {
                p.fill_grad(0.0f);
            } catch (...) {
                // If gradient not allocated, that's fine for zero_grad
            }
        }
    }
}

Tensor Module::operator()(const Tensor& input) {
    return forward(input);
}

void Module::register_parameter(Tensor p) {
    params_.push_back(p);
}

// ============================================================================
// Linear
// ============================================================================

Linear::Linear(int in_features, int out_features, bool use_bias) {
    TensorOptions opts = TensorOptions().with_req_grad(true);
    
    // Initialize weights with He/Kaiming initialization basic equivalent
    // scaling by 1/sqrt(fan_in) for uniform or normal
    float stdv = 1.0f / std::sqrt(static_cast<float>(in_features));
    
    weight = Tensor::randn<float>(Shape{{in_features, out_features}}, opts, 1.0f) * stdv;     
    
    if (use_bias) {
        bias = Tensor::zeros(Shape{{out_features}}, opts);
    } // else we should handle no bias case, but for now assuming always bias or zero tensor
      // If no bias, we could use empty tensor? matrix add supports it?
      // For simplicity, if no bias, we just init to zeros with requires_grad=false?
      // Or 0s.
    
    register_parameter(weight);
    if (use_bias) {
        register_parameter(bias);
    }
}

Tensor Linear::forward(const Tensor& input) {
    // y = x @ W + b
    Tensor z = autograd::matmul(input, weight);
    if (bias.is_valid()) {
        return autograd::add(z, bias);
    }
    return z;
}

std::vector<Tensor> Linear::parameters() {
    std::vector<Tensor> p = {weight};
    if (bias.is_valid()) p.push_back(bias);
    return p;
}

void Linear::to(DeviceIndex dev) {
    weight = weight.to(dev);
    if (bias.is_valid()) bias = bias.to(dev);
}

// ============================================================================
// ReLU
// ============================================================================

Tensor ReLU::forward(const Tensor& input) {
    return autograd::relu(input);
}

// ============================================================================
// Embedding
// ============================================================================

Embedding::Embedding(int num_embeddings, int embedding_dim, int padding_idx) 
    : padding_idx(padding_idx) {
    TensorOptions opts = TensorOptions().with_req_grad(true);
    
    // Normal distribution initialization (small normal)
    weight = Tensor::randn<float>(Shape{{num_embeddings, embedding_dim}}, opts, 0.02f);
    
    // Zero out padding row if requested
    if (padding_idx >= 0 && padding_idx < num_embeddings) {
        float* w_ptr = weight.data<float>();
        std::fill(w_ptr + (size_t)padding_idx * embedding_dim, 
                  w_ptr + (size_t)(padding_idx + 1) * embedding_dim, 0.0f);
    }
    
    register_parameter(weight);
}

Tensor Embedding::forward(const Tensor& input) {
    return autograd::embedding(weight, input);
}

std::vector<Tensor> Embedding::parameters() {
    return {weight};
}

void Embedding::to(DeviceIndex dev) {
    weight = weight.to(dev);
}

// ============================================================================
// Sequential
// ============================================================================

Sequential::Sequential(std::initializer_list<Module*> modules) {
    for (auto* m : modules) {
        add(std::shared_ptr<Module>(m));
    }
}

void Sequential::add(std::shared_ptr<Module> module) {
    modules_.push_back(module);
    // REMOVED: stale parameter flattening here. 
    // Sequential::parameters() is recursive and will find them dynamically.
}

Tensor Sequential::forward(const Tensor& input) {
    Tensor x = input;
    for (auto& m : modules_) {
        x = m->forward(x);
    }
    return x;
}

std::vector<Tensor> Sequential::parameters() {
    std::vector<Tensor> all_params;
    for (auto& m : modules_) {
        auto sub_params = m->parameters();
        all_params.insert(all_params.end(), sub_params.begin(), sub_params.end());
    }
    return all_params;
}

void Sequential::to(DeviceIndex dev) {
    for (auto& m : modules_) {
        m->to(dev);
    }
}

// ============================================================================
// Loss Functions
// ============================================================================

Tensor mse_loss(const Tensor& pred, const Tensor& target) {
    // loss = mean((pred - target)^2)
    Tensor neg_target = target * -1.0f;
    Tensor diff = autograd::add(pred, neg_target);
    Tensor sq_diff = autograd::mul(diff, diff);
    return autograd::mean(sq_diff);
}

} // namespace nn
} // namespace OwnTensor