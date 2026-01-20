#include "core/TensorImpl.h"
#include "core/Tensor.h"
#include "core/Views/ViewUtils.h"
#include <stdexcept>

namespace OwnTensor {

// ============================================================================
// Constructors
// ============================================================================

TensorImpl::TensorImpl(Storage&& storage,
                       const Shape& shape,
                       const Stride& stride,
                       int64_t offset,
                       Dtype dtype,
                       DeviceIndex device,
                       intrusive_ptr<TensorImpl> base_impl)
    : storage_(std::move(storage)),
      shape_(shape),
      stride_(stride),
      storage_offset_(offset),
      dtype_(dtype),
      device_(device),
      base_impl_(std::move(base_impl)) {
    // autograd_meta_ is nullptr - lazy allocation
}

TensorImpl::TensorImpl(const Shape& shape,
                       Dtype dtype,
                       DeviceIndex device,
                       bool requires_grad)
    : shape_(shape),
      dtype_(dtype),
      device_(device),
      storage_offset_(0) {
    
    // Calculate storage size
    size_t elem_count = 1;
    for (auto dim : shape.dims) {
        elem_count *= dim;
    }
    
    size_t elem_size = Tensor::dtype_size(dtype);
    size_t nbytes = elem_count * elem_size;
    
    // Align memory for performance
    if (device.is_cpu()) {
        nbytes = (nbytes + 63) & ~63;  // 64-byte alignment for CPU
    } else {
        nbytes = ((nbytes + 256 - 1) / 256) * 256;  // 256-byte alignment for GPU
    }
    
    // Create storage
    storage_ = Storage(nbytes, dtype, device, nullptr);
    
    // Compute strides
    stride_ = ViewUtils::compute_strides(shape);
    
    // Create autograd metadata if requires_grad
    if (requires_grad) {
        autograd_meta_ = std::make_unique<AutogradMeta>(true);
    }
}

TensorImpl::~TensorImpl() {
    release_resources();
}

// ============================================================================
// Metadata Accessors
// ============================================================================

size_t TensorImpl::numel() const {
    size_t total = 1;
    for (auto dim : shape_.dims) {
        total *= dim;
    }
    return total;
}

size_t TensorImpl::nbytes() const {
    return numel() * Tensor::dtype_size(dtype_);
}

int64_t TensorImpl::ndim() const {
    return static_cast<int64_t>(shape_.dims.size()); //can use size_t and remove static cast
}

// ============================================================================
// Data Access
// ============================================================================

void* TensorImpl::mutable_data() {
    uint8_t* base_ptr = storage_.data_ptr();
    if (!base_ptr) {
        return nullptr;
    }
    
    // Apply storage offset
    size_t elem_size = Tensor::dtype_size(dtype_);
    return base_ptr + (storage_offset_ * elem_size);
}

const void* TensorImpl::data() const {
    const uint8_t* base_ptr = storage_.data_ptr();
    if (!base_ptr) {
        return nullptr;
    }
    
    // Apply storage offset
    size_t elem_size = Tensor::dtype_size(dtype_);
    return base_ptr + (storage_offset_ * elem_size);
}

// ============================================================================
// Autograd Methods
// ============================================================================

void TensorImpl::set_requires_grad(bool requires_grad) {
    if (requires_grad && !autograd_meta_) {
        // Lazy allocation of autograd metadata - this happens if the tensor created with autograd_meta_ = false
        autograd_meta_ = std::make_unique<AutogradMeta>(true);
    } else if (autograd_meta_) {
        autograd_meta_->set_requires_grad(requires_grad, this);
    }
}

bool TensorImpl::requires_grad() const {
    return autograd_meta_ && autograd_meta_->requires_grad();
}

Tensor& TensorImpl::mutable_grad() {
    if (!autograd_meta_) {
        throw std::runtime_error("TensorImpl::mutable_grad: tensor does not require gradients");
    }
    return autograd_meta_->mutable_grad(this);
}

const Tensor& TensorImpl::grad() const {
    if (!autograd_meta_) {
        throw std::runtime_error("TensorImpl::grad: tensor does not require gradients");
    }
    return autograd_meta_->grad();
}

void TensorImpl::set_autograd_meta(std::unique_ptr<AutogradMetaInterface> autograd_meta) {
    autograd_meta_ = std::move(autograd_meta);
}
void TensorImpl::zero_grad() {
    if (autograd_meta_ && autograd_meta_->has_grad()) {
        // Zero out the gradient by getting mutable reference and filling with zeros
        Tensor& grad_tensor = autograd_meta_->mutable_grad(this);
        // Use fill method to zero the gradient
        grad_tensor.fill<float>(0.0f);
    }
}
// ============================================================================
// Metadata Mutation
// ============================================================================

void TensorImpl::set_sizes_and_strides(const Shape& new_shape, const Stride& new_stride) {
    shape_ = new_shape;
    stride_ = new_stride;
}

// ============================================================================
// Cleanup
// ============================================================================

void TensorImpl::release_resources() {
    // Storage will be automatically cleaned up by its destructor
    // AutogradMeta will be automatically cleaned up by unique_ptr
}

} // namespace OwnTensor
