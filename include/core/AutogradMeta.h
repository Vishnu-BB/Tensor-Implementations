#pragma once

#include <memory>
#include <mutex>

namespace OwnTensor {

// Forward declarations
class Tensor;
class TensorImpl;

/**
 * AutogradMeta stores autograd-specific metadata for a tensor.
 * This is separate from TensorImpl to keep autograd concerns isolated.
 * 
 * AutogradMeta is responsible for:
 * - Storing the gradient tensor
 * - Tracking requires_grad flag
 * - Thread-safe gradient access
 * - (Future) Gradient function management
 */
class AutogradMeta {
private:
    std::unique_ptr<Tensor> grad_;     // Gradient tensor (lazy allocated)
    bool requires_grad_;                // Whether this tensor requires gradients
    mutable std::mutex mutex_;          // Mutex for thread-safe access
    
    // Placeholders for future autograd integration
    // std::shared_ptr<Node> grad_fn_;            // Gradient function
    // std::weak_ptr<Node> grad_accumulator_;     // Gradient accumulator
    // uint32_t output_nr_ = 0;                   // Output number in gradient edge

public:
    // ========================================================================
    // Constructors
    // ========================================================================
    
    /**
     * Constructor
     * @param requires_grad Whether gradients should be tracked
     */
    explicit AutogradMeta(bool requires_grad = false);
    
    // Destructor
    ~AutogradMeta() = default;
    
    // Move semantics
    AutogradMeta(AutogradMeta&& other) noexcept;
    AutogradMeta& operator=(AutogradMeta&& other) noexcept;
    
    // No copy (autograd metadata should not be copied)
    AutogradMeta(const AutogradMeta&) = delete;
    AutogradMeta& operator=(const AutogradMeta&) = delete;
    
    // ========================================================================
    // Gradient Access
    // ========================================================================
    
    /**
     * Get mutable reference to gradient tensor
     * Allocates gradient if it doesn't exist
     * @param self_impl Pointer to owning TensorImpl (for lazy allocation)
     * @return Reference to gradient tensor
     */
    Tensor& mutable_grad(TensorImpl* self_impl);
    
    /**
     * Get const reference to gradient tensor
     * Throws if gradient doesn't exist
     * @return Const reference to gradient tensor
     */
    const Tensor& grad() const;
    
    /**
     * Set gradient tensor
     * @param new_grad New gradient tensor
     */
    void set_grad(const Tensor& new_grad);
    
    /**
     * Check if gradient exists
     * @return true if gradient has been allocated
     */
    bool has_grad() const;
    
    /**
     * Reset gradient to uninitialized state
     */
    void reset_grad();
    
    // ========================================================================
    // Requires Grad
    // ========================================================================
    
    /**
     * Set requires_grad flag
     * @param requires_grad New value for requires_grad
     */
    void set_requires_grad(bool requires_grad);
    
    /**
     * Get requires_grad flag
     * @return true if tensor requires gradients
     */
    bool requires_grad() const { return requires_grad_; }
    
    // ========================================================================
    // Future: Gradient Function Management
    // ========================================================================
    
    // void set_grad_fn(std::shared_ptr<Node> grad_fn);
    // std::shared_ptr<Node> grad_fn() const;
    // void set_grad_accumulator(std::weak_ptr<Node> grad_accumulator);
    // std::shared_ptr<Node> grad_accumulator() const;
};

} // namespace OwnTensor
