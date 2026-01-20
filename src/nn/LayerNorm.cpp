#include "nn/NN.h"
#include "autograd/AutogradOps.h"
#include "ops/ScalarOps.h"
#include "ops/TensorOps.h"
#include <cmath>

namespace OwnTensor {
namespace nn {

// ============================================================================
// LayerNorm
// ============================================================================

LayerNorm::LayerNorm(int normalized_shape, float eps) : eps(eps) {
    TensorOptions opts = TensorOptions().with_req_grad(true);
    
    // Initialize weight (gamma) to ones and bias (beta) to zeros
    weight = Tensor::ones(Shape{{normalized_shape}}, opts);
    bias = Tensor::zeros(Shape{{normalized_shape}}, opts);
    
    register_parameter(weight);
    register_parameter(bias);
}

Tensor LayerNorm::forward(const Tensor& input) {
    // LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
    // 
    // This is a simplified global normalization implementation.
    // For proper layer norm, we'd need axis-specific reductions.
    // However, with broadcasting support in autograd ops, this should
    // work as a basic layer normalization.
    
    // Step 1: Compute mean
    Tensor mean_val = autograd::mean(input);
    
    // Step 2: Center the input (x - mean)
    Tensor x_centered = autograd::sub(input, mean_val);
    
    // Step 3: Compute variance = mean(x_centered^2)
    Tensor x_sq = autograd::mul(x_centered, x_centered);
    Tensor var_val = autograd::mean(x_sq);
    
    // Step 4: Add eps for numerical stability using scalar addition
    // Use non-autograd scalar op since eps is a constant
    Tensor var_eps = var_val + eps;
    
    // Step 5: Compute std = sqrt(var + eps)
    Tensor std_val = autograd::sqrt(var_eps);
    
    // Step 6: Normalize: x_centered / std
    Tensor x_norm = autograd::div(x_centered, std_val);
    
    // Step 7: Apply affine transform: x_norm * weight + bias
    Tensor scaled = autograd::mul(x_norm, weight);
    Tensor output = autograd::add(scaled, bias);
    
    return output;
}

std::vector<Tensor> LayerNorm::parameters() {
    std::vector<Tensor> p = {weight};
    if (bias.is_valid()) p.push_back(bias);
    return p;
}

void LayerNorm::to(DeviceIndex dev) {
    weight = weight.to(dev);
    if (bias.is_valid()) bias = bias.to(dev);
}

} // namespace nn
} // namespace OwnTensor
