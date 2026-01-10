#pragma once

/**
 * @file CppNode.h
 * @brief Base class for custom C++ autograd functions.
 * 
 * Provides the template for creating custom backward functions
 * similar to PyTorch's torch.autograd.Function.
 */

#include "autograd/Node.h"
#include "autograd/AutogradContext.h"
#include <memory>

namespace OwnTensor {
namespace autograd {

/**
 * @brief Base template for custom autograd functions.
 * 
 * ## Usage
 * ```cpp
 * class MyOp : public CppNode<MyOp> {
 * public:
 *     // Forward pass (static method)
 *     static variable_list forward(AutogradContext* ctx,
 *                                   const Tensor& a, const Tensor& b) {
 *         ctx->save_for_backward({a, b});
 *         return {a * b};
 *     }
 *     
 *     // Backward pass (static method)
 *     static variable_list backward(AutogradContext* ctx,
 *                                    const variable_list& grad_outputs) {
 *         auto saved = ctx->get_saved_variables();
 *         return {grad_outputs[0] * saved[1],
 *                 grad_outputs[0] * saved[0]};
 *     }
 * };
 * 
 * // Usage:
 * auto result = MyOp::apply(a, b);
 * ```
 */
template<typename T>
class CppNode : public Node {
protected:
    AutogradContext ctx_;
    
public:
    CppNode() : Node(0) {
        ctx_.set_grad_fn(this->shared_from_this());
    }
    
    std::string name() const override {
        return "CppNode";  // Subclasses can override
    }
    
    variable_list apply(variable_list&& grads) override {
        return T::backward(&ctx_, grads);
    }
    
    /**
     * @brief Apply the function with gradient tracking.
     * 
     * This is the main entry point for using the custom function.
     */
    template<typename... Args>
    static variable_list apply(Args&&... args) {
        // Create node
        auto node = std::make_shared<T>();
        
        // Call forward
        variable_list outputs = T::forward(&node->ctx_, std::forward<Args>(args)...);
        
        // Set grad_fn on outputs
        for (size_t i = 0; i < outputs.size(); ++i) {
            if (outputs[i].requires_grad()) {
                outputs[i].set_grad_fn(node);
            }
        }
        
        return outputs;
    }
    
    /**
     * @brief Release saved variables after backward.
     */
    void release_variables() {
        ctx_.release_variables();
    }
};

} // namespace autograd
} // namespace OwnTensor
