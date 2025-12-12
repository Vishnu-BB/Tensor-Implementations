#pragma once

#ifndef TENSOR_DISPATCH_H
#define TENSOR_DISPATCH_H

// ✅ CRITICAL: Include Dtype.h first for enum definition
#include "dtype/Dtype.h"
#include <stdexcept>
#include <cstdint>

#ifdef __CUDACC__
    #include <cuda_fp16.h>
    #include <cuda_bf16.h>
    #include <cuda_fp8.h>  // FP8 native types (CUDA 11.8+)
#endif
#include "dtype/Types.h"

// ✅ Forward declare the Tensor class (avoid circular dependency)
namespace OwnTensor {
    class Tensor;
}

// ✅ CRITICAL FIX: Manually specify the type mapping WITHOUT dtype_traits
// This avoids the circular dependency and NVCC template resolution issues
namespace OwnTensor {

// ✅ Simple type resolver that works in both CPU and CUDA contexts
template<Dtype dt> struct DtypeToType;

// Integer types
template<> struct DtypeToType<Dtype::Int8> { using type = int8_t; };
template<> struct DtypeToType<Dtype::Int16>  { using type = int16_t; };
template<> struct DtypeToType<Dtype::Int32>  { using type = int32_t; };
template<> struct DtypeToType<Dtype::Int64>  { using type = int64_t; };
//Unsigned Integer types
template<> struct DtypeToType<Dtype::UInt8> { using type = uint8_t;};
template<> struct DtypeToType<Dtype::UInt16> { using type = uint16_t;};
template<> struct DtypeToType<Dtype::UInt32> { using type = uint32_t;};
template<> struct DtypeToType<Dtype::UInt64> { using type = uint64_t;};
// Standard floating point
template<> struct DtypeToType<Dtype::Float32> { using type = float; };
template<> struct DtypeToType<Dtype::Float64> { using type = double; };

//Boolean type
template<> struct DtypeToType<Dtype::Bool> { using type = bool;};

// ✅ Half precision types - resolve based on compilation context
#ifdef __CUDACC__
    // CUDA compilation - use native CUDA types
    template<> struct DtypeToType<Dtype::Float16>  { using type = __half; };
    template<> struct DtypeToType<Dtype::Bfloat16> { using type = __nv_bfloat16; };
    template<> struct DtypeToType<Dtype::Float8_E4M3FN> { using type = __nv_fp8_e4m3; };
    template<> struct DtypeToType<Dtype::Float8_E5M2> { using type = __nv_fp8_e5m2; };
    template<> struct DtypeToType<Dtype::Complex32> { using type = complex32_t; };
    template<> struct DtypeToType<Dtype::Complex64> { using type = complex64_t; };
    template<> struct DtypeToType<Dtype::Complex128> { using type = complex128_t; };
#else
    // CPU compilation - use custom types
    template<> struct DtypeToType<Dtype::Float16>  { using type = float16_t; };
    template<> struct DtypeToType<Dtype::Bfloat16> { using type = bfloat16_t; };
    template<> struct DtypeToType<Dtype::Float8_E4M3FN> { using type = float8_e4m3fn_t; };
    template<> struct DtypeToType<Dtype::Float8_E5M2> { using type = float8_e5m2_t; };
    template<> struct DtypeToType<Dtype::Complex32> { using type = complex32_t; };
    template<> struct DtypeToType<Dtype::Complex64> { using type = complex64_t; };
    template<> struct DtypeToType<Dtype::Complex128> { using type = complex128_t; };
#endif

// ✅ Runtime dispatcher using the simple type resolver
template<typename Func>
static auto dispatch_by_dtype(Dtype dtype, Func&& f) {
    switch(dtype) {
        case Dtype::Int16:    return f(typename DtypeToType<Dtype::Int16>::type{});
        case Dtype::Int32:    return f(typename DtypeToType<Dtype::Int32>::type{});
        case Dtype::Int64:    return f(typename DtypeToType<Dtype::Int64>::type{});
        case Dtype::Float32:  return f(typename DtypeToType<Dtype::Float32>::type{});
        case Dtype::Float64:  return f(typename DtypeToType<Dtype::Float64>::type{});
        case Dtype::Bfloat16: return f(typename DtypeToType<Dtype::Bfloat16>::type{});
        case Dtype::Float16:  return f(typename DtypeToType<Dtype::Float16>::type{});
        case Dtype::Float8_E4M3FN: return f(typename DtypeToType<Dtype::Float8_E4M3FN>::type{});
        case Dtype::Float8_E5M2:   return f(typename DtypeToType<Dtype::Float8_E5M2>::type{});
        case Dtype::Bool:   return f(typename DtypeToType<Dtype::Bool>::type{});
        case Dtype::UInt8: return f(typename DtypeToType<Dtype::UInt8>::type{});
        case Dtype::UInt16: return f(typename DtypeToType<Dtype::UInt16>::type{});
        case Dtype::UInt32: return f(typename DtypeToType<Dtype::UInt32>::type{});
        case Dtype::UInt64: return f(typename DtypeToType<Dtype::UInt64>::type{});
        case Dtype::Complex32: return f(typename DtypeToType<Dtype::Complex32>::type{});
        case Dtype::Complex64: return f(typename DtypeToType<Dtype::Complex64>::type{});
        case Dtype::Complex128: return f(typename DtypeToType<Dtype::Complex128>::type{});
        default:
            throw std::runtime_error("Unsupported Dtype");
    }
}

} // namespace OwnTensor

#endif