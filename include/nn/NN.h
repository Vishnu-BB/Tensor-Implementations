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
    
    // Move module parameters to device
    virtual void to(DeviceIndex dev);
    
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
    std::vector<Tensor> parameters() override;
    void to(DeviceIndex dev) override;
};

class ReLU : public Module {
public:
    Tensor forward(const Tensor& input) override;
};

class Embedding : public Module {
public:
    Tensor weight;
    int padding_idx;
    
    Embedding(int num_embeddings, int embedding_dim, int padding_idx = -1);
    
    Tensor forward(const Tensor& input) override;
    std::vector<Tensor> parameters() override;
    void to(DeviceIndex dev) override;
};

class LayerNorm : public Module {
public:
    Tensor weight;
    Tensor bias;
    float eps;
    
    LayerNorm(int normalized_shape, float eps = 1e-5);
    
    Tensor forward(const Tensor& input) override;
    std::vector<Tensor> parameters() override;
    void to(DeviceIndex dev) override;
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
    std::vector<Tensor> parameters() override;
    void to(DeviceIndex dev) override;
};

// ============================================================================
// Loss Functions
// ============================================================================

Tensor mse_loss(const Tensor& pred, const Tensor& target);

} // namespace nn
} // namespace OwnTensor
