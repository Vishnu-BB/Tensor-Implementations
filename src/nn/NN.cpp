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

void Module::zero_grad() {
    for (auto& p : params_) {
        // Only attempt to zero gradients if they require grad and exist
        if (p.requires_grad() && p.grad() != nullptr) {
            p.fill_grad(0.0f);
        }
    }
}

void Module::to(DeviceIndex dev) {
    for (auto& p : params_) {
        if (p.is_valid() && p.device() != dev) {
            // Create new tensor on target device with same properties
            TensorOptions opts = TensorOptions()
                .with_dtype(p.dtype())
                .with_device(dev)
                .with_req_grad(p.requires_grad());
            
            Tensor new_tensor(p.shape(), opts);
            
            // Copy data from old tensor to new tensor
            size_t num_bytes = p.numel() * Tensor::dtype_size(p.dtype());
            std::memcpy(new_tensor.data(), p.data(), num_bytes);
            
            // Replace the parameter
            p = new_tensor;
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

// ============================================================================
// ReLU
// ============================================================================

Tensor ReLU::forward(const Tensor& input) {
    return autograd::relu(input);
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
    
    // Register parameters from submodule
    auto sub_params = module->parameters();
    // Insert parameters into our params list so optimizer can see them easily
    // Note: This flattens parameters. If submodules change params later, this won't track.
    // Ideally parameters() should always recurse.
    // But for this simple implementation, copying is okay if structure is static.
    // A better implementation of parameters() would be recursive.
    // Let's make parameters() recursive instead of registering?
    // But then register_parameter needs to store locally.
    // Let's stick to the flattening for now as per test_mlp_lib.cpp logic
    params_.insert(params_.end(), sub_params.begin(), sub_params.end());
}

Tensor Sequential::forward(const Tensor& input) {
    Tensor x = input;
    for (auto& m : modules_) {
        x = m->forward(x);
    }
    return x;
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
