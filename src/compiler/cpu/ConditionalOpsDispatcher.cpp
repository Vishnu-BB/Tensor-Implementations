#include <stdexcept>
#include "core/Tensor.h"
#include "ops/helpers/ConditionalOps.h"
#include "ops/helpers/BroadcastUtils.h"  // ✅ For broadcast_rhs_to_lhs
#include "core/TensorDispatch.h"
#include "dtype/Types.h"
#include "dtype/DtypeTraits.h"  // ✅ For type_to_dtype and promote_dtypes_bool
using namespace std;
namespace OwnTensor {

// Backend declarations
void cpu_where(const Tensor& condition, const Tensor& input, 
               const Tensor& other, Tensor& out);
void cuda_where(const Tensor& condition, const Tensor& input,
                const Tensor& other, Tensor& out);

// Main where implementation
    Tensor where(const Tensor& condition, const Tensor& input, const Tensor& other) {
    // Validate condition dtype
    if (condition.dtype() != Dtype::Bool) {
        throw std::runtime_error("Condition must be Bool dtype");
    }
    
    // Validate all tensors on same device
    if (condition.device().index != input.device().index || input.device().index != other.device().index) {
        throw std::runtime_error("All tensors must be on the same device");
    }
    
    // ✅ Determine output dtype (promote input and other)
    Dtype output_dtype = promote_dtypes_bool(input.dtype(), other.dtype());
    
    // ✅ Compute output shape (broadcasting)
    Shape output_shape = condition.shape();
    if (condition.shape().dims != input.shape().dims || condition.shape().dims != other.shape().dims) {
        // Need to broadcast - compute the broadcasted shape
        std::vector<int64_t> temp_shape = broadcast_shape(condition.shape().dims, input.shape().dims);
        temp_shape = broadcast_shape(temp_shape, other.shape().dims);
        output_shape = Shape{temp_shape};
    }
    
    // Create output tensor
    Tensor out(output_shape, output_dtype, input.device(), false);
    
    // Dispatch to backend
    if (condition.device().is_cuda()) {
        cuda_where(condition, input, other, out);
    } else {
        cpu_where(condition, input, other, out);
    }
    
    return out;
}

// Scalar overloads - create full tensors and call main function
template <typename T>
Tensor where(const Tensor& condition, T input_scalar, const Tensor& other) {
    // ✅ 1. Check if other tensor can be broadcasted to condition shape
    Shape output_shape = Shape{broadcast_rhs_to_lhs(condition.shape().dims, other.shape().dims)};
    
    // ✅ 2. Get dtype from scalar type T
    Dtype scalar_dtype = type_to_dtype<T>();
    
    // ✅ 3. Promote scalar dtype with tensor dtype
    Dtype promoted_dtype = promote_dtypes_bool(scalar_dtype, other.dtype());
    
    // ✅ 4. Promote other tensor if needed (no copy, just type conversion)
    Tensor other_promoted = (other.dtype() != promoted_dtype) ? other.as_type(promoted_dtype) : other;
    
    // ✅ 5. Create output tensor
    Tensor out(output_shape, promoted_dtype, condition.device(), false);
    
    // ✅ 6. Call scalar backend directly - NO tensor creation for scalar!
    if (condition.device().is_cuda()) {
        cuda_where_scalar_tensor(condition, input_scalar, other_promoted, out);
    } else {
        cpu_where_scalar_tensor(condition, input_scalar, other_promoted, out);
    }
    
    return out;
}

template <typename T>
Tensor where(const Tensor& condition, const Tensor& input, T other_scalar) {
    // ✅ 1. Check if input tensor can be broadcasted to condition shape
    Shape output_shape = Shape{broadcast_rhs_to_lhs(condition.shape().dims, input.shape().dims)};
    
    // ✅ 2. Get dtype from scalar type T
    Dtype scalar_dtype = type_to_dtype<T>();
    
    // ✅ 3. Promote scalar dtype with tensor dtype
    Dtype promoted_dtype = promote_dtypes_bool(scalar_dtype, input.dtype());
    
    // ✅ 4. Promote input tensor if needed (no copy, just type conversion)  
    Tensor input_promoted = (input.dtype() != promoted_dtype) ? input.as_type(promoted_dtype) : input;
    
    // ✅ 5. Create output tensor
    Tensor out(output_shape, promoted_dtype, condition.device(), false);
    
    // ✅ 6. Call scalar backend directly - NO tensor creation for scalar!
    if (condition.device().is_cuda()) {
        cuda_where_tensor_scalar(condition, input_promoted, other_scalar, out);
    } else {
        cpu_where_tensor_scalar(condition, input_promoted, other_scalar, out);
    }
    
    return out;
}

// Two-template version handles BOTH same-type and mixed-type scalars
template <typename T, typename U>
Tensor where(const Tensor& condition, T input_scalar, U other_scalar) {
    // ✅ 1. Get dtypes from both scalar types
    Dtype scalar_dtype1 = type_to_dtype<T>();
    Dtype scalar_dtype2 = type_to_dtype<U>();
    
    // ✅ 2. Promote the two scalar types (handles same-type correctly)
    Dtype output_dtype = promote_dtypes_bool(scalar_dtype1, scalar_dtype2);
    
    // ✅ 3. Create output tensor
    Tensor out(condition.shape(), output_dtype, condition.device(), false);
    
    // ✅ 4. Call scalar backend directly - NO tensor creation for scalars!
    if (condition.device().is_cuda()) {
        cuda_where_scalar_scalar(condition, input_scalar, other_scalar, out);
    } else {
        cpu_where_scalar_scalar(condition, input_scalar, other_scalar, out);
    }
    
    return out;
}



template Tensor where<int>(const Tensor& condition,const Tensor& input, int inp_scalar );
template Tensor where<float>(const Tensor& condition,const Tensor& input, float inp_scalar );
template Tensor where<double>(const Tensor& condition,const Tensor& input, double inp_scalar );
template Tensor where<long>(const Tensor& condition,const Tensor& input, long inp_scalar );

template Tensor where<int>(const Tensor& condition, int inp_scalar ,const Tensor& input);
template Tensor where<float>(const Tensor& condition, float inp_scalar ,const Tensor& input);
template Tensor where<double>(const Tensor& condition, double inp_scalar,const Tensor& input );
template Tensor where<long>(const Tensor& condition, long inp_scalar,const Tensor& input );

// Template instantiations - all use two template parameters
// Same-type cases
template Tensor where<int, int>(const Tensor& condition, int inp_scalar, int other_scalar);
template Tensor where<float, float>(const Tensor& condition, float inp_scalar, float other_scalar);
template Tensor where<double, double>(const Tensor& condition, double inp_scalar, double other_scalar);
template Tensor where<long, long>(const Tensor& condition, long inp_scalar, long other_scalar);

// Mixed-type cases (add more combinations as needed)
template Tensor where<int, float>(const Tensor& condition, int inp_scalar, float other_scalar);
template Tensor where<int, double>(const Tensor& condition, int inp_scalar, double other_scalar);
template Tensor where<int, long>(const Tensor& condition, int inp_scalar, long other_scalar);

template Tensor where<float, int>(const Tensor& condition, float inp_scalar, int other_scalar);
template Tensor where<float, double>(const Tensor& condition, float inp_scalar, double other_scalar);
template Tensor where<float, long>(const Tensor& condition, float inp_scalar, long other_scalar);

template Tensor where<long, int>(const Tensor& condition, long inp_scalar, int other_scalar);
template Tensor where<long, float>(const Tensor& condition, long inp_scalar, float other_scalar);
template Tensor where<long, double>(const Tensor& condition, long inp_scalar, double other_scalar);

template Tensor where<double, int>(const Tensor& condition, double inp_scalar, int other_scalar);
template Tensor where<double, float>(const Tensor& condition, double inp_scalar, float other_scalar);
template Tensor where<double, long>(const Tensor& condition, double inp_scalar, long other_scalar);



} // namespace OwnTensor
