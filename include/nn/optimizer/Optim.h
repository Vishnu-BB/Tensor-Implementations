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

// *********************************************************************************************
// ================================ Adam Optimizer =============================================
// *********************************************************************************************

/**
 * @brief Adam optimizer with momentum and adaptive learning rates
 * 
 * Implements the Adam optimization algorithm:
 * m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
 * v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
 * m_hat = m_t / (1 - beta1^t)
 * v_hat = v_t / (1 - beta2^t)
 * theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)
 */
class Adam {
private:
    float lr_;
    float beta1_;
    float beta2_;
    float eps_;
    float weight_decay_;
    int64_t step_count_;
    
    // First moment estimates (momentum)
    std::vector<Tensor> m_;
    // Second moment estimates (RMSProp)
    std::vector<Tensor> v_;
    // Reference to parameters
    std::vector<Tensor*> params_;
    bool initialized_;
    
public:
    /**
     * @brief Construct Adam optimizer
     * 
     * @param params Vector of parameter tensors to optimize
     * @param lr Learning rate (default: 0.001)
     * @param beta1 Exponential decay rate for first moment (default: 0.9)
     * @param beta2 Exponential decay rate for second moment (default: 0.999)
     * @param eps Small constant for numerical stability (default: 1e-8)
     * @param weight_decay L2 regularization coefficient (default: 0)
     */
    Adam(std::vector<Tensor*> params, 
         float lr = 0.001f, 
         float beta1 = 0.9f, 
         float beta2 = 0.999f,
         float eps = 1e-8f,
         float weight_decay = 0.0f);
    
    /**
     * @brief Perform a single optimization step
     */
    void step();
    
    /**
     * @brief Zero all parameter gradients
     */
    void zero_grad();
    
    /**
     * @brief Get current learning rate
     */
    float get_lr() const { return lr_; }
    
    /**
     * @brief Set learning rate
     */
    void set_lr(float lr) { lr_ = lr; }
    
    /**
     * @brief Get step count
     */
    int64_t get_step_count() const { return step_count_; }
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


/**
 * @brief Clip gradients by global norm
 * 
 * Rescales gradients so that their global norm does not exceed max_norm.
 * 
 * @param params Vector of parameter tensors
 * @param max_norm Maximum allowed gradient norm
 * @return The original global norm before clipping
 */
float clip_grad_norm_(std::vector<Tensor*>& params, float max_norm);
} // namespace OwnTensor
