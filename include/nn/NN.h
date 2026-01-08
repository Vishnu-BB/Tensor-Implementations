#pragma once

#include "core/Tensor.h"
#include <vector>
#include <memory>
#include <initializer_list>

namespace OwnTensor {
namespace nn {

// ============================================================================
// Base Module
// ============================================================================

class Module {
public:
    virtual ~Module() = default;
    
    // Forward pass
    virtual Tensor forward(const Tensor& input) = 0;
    
    // Get parameters
    virtual std::vector<Tensor> parameters();
    
    // Zero gradients
    void zero_grad();
    
    // Operator() alias for forward
    Tensor operator()(const Tensor& input);
    
protected:
    std::vector<Tensor> params_;
    
    void register_parameter(Tensor p);
};

// ============================================================================
// Layers
// ============================================================================

class Linear : public Module {
public:
    Tensor weight;
    Tensor bias;
    
    Linear(int in_features, int out_features, bool bias = true);
    
    Tensor forward(const Tensor& input) override;
};

class ReLU : public Module {
public:
    Tensor forward(const Tensor& input) override;
};

// ============================================================================
// Containers
// ============================================================================

class Sequential : public Module {
private:
    std::vector<std::shared_ptr<Module>> modules_;
    
public:
    Sequential(std::initializer_list<Module*> modules);
    
    // Templated add for building incrementally?
    void add(std::shared_ptr<Module> module);
    
    Tensor forward(const Tensor& input) override;
};

// ============================================================================
// Loss Functions
// ============================================================================

Tensor mse_loss(const Tensor& pred, const Tensor& target);

} // namespace nn
} // namespace OwnTensor
