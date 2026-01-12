#pragma once

#include "core/Tensor.h"
#include <vector>
#include <unordered_map>
#include <memory>

namespace OwnTensor {
namespace nn {

class Optimizer {
public:
    Optimizer(const std::vector<Tensor>& params);
    virtual ~Optimizer() = default;

    virtual void step() = 0;
    void zero_grad();
    
    // For mixed precision: stores master copy of parameters in Float32
    Tensor* get_master_weight(const Tensor& v);

protected:
    std::vector<Tensor> params_;
    // Map from Tensor source pointer to master copy. 
    // We use raw pointer to TensorImpl or some unique ID as key.
    // Since Tensors are shared_ptr-like, we use the impl pointer.
    std::unordered_map<void*, Tensor> master_params_; 
};

class SGDOptimizer : public Optimizer {
public:
    SGDOptimizer(const std::vector<Tensor>& params, float learning_rate = 0.01);
    
    void step() override;

private:
    float learning_rate_;
};

class Adam : public Optimizer {
public:
    Adam(const std::vector<Tensor>& params, float alpha = 0.001, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8);
    
    void step() override;

private:
    float alpha_;
    float beta1_;
    float beta2_;
    float epsilon_;
    int t_;

    std::unordered_map<void*, Tensor> m_;             // First moment
    std::unordered_map<void*, Tensor> v_;             // Second moment
};

class AdamW : public Optimizer {
public:
   AdamW(const std::vector<Tensor>& params, float alpha = 0.001, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8, float weight_decay=1e-2);

   void step() override;

private:
   float alpha_;
   float beta1_;
   float beta2_;
   float epsilon_;
   float weight_decay_;
   int t_;

   std::unordered_map<void*, Tensor> m_;             // First moment
   std::unordered_map<void*, Tensor> v_;             // Second moment
};

} // namespace nn
} // namespace OwnTensor
