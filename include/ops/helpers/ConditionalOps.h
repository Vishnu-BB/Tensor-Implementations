#pragma once
#ifndef CONDITIONAL_OPS_H
#define CONDITIONAL_OPS_H

#include "core/Tensor.h"

namespace OwnTensor {

// Forward declarations for CPU and CUDA backends - Tensor variants
void cpu_where(const Tensor& condition, const Tensor& input, 
               const Tensor& other, Tensor& out);

void cuda_where(const Tensor& condition, const Tensor& input,
                const Tensor& other, Tensor& out);

// Scalar backend variants - CPU
template<typename T>
void cpu_where_scalar_tensor(const Tensor& condition, T input_scalar, 
                              const Tensor& other, Tensor& out);

template<typename T>
void cpu_where_tensor_scalar(const Tensor& condition, const Tensor& input, 
                              T other_scalar, Tensor& out);

template<typename T, typename U>
void cpu_where_scalar_scalar(const Tensor& condition, T input_scalar, 
                              U other_scalar, Tensor& out);

// Scalar backend variants - CUDA
template<typename T>
void cuda_where_scalar_tensor(const Tensor& condition, T input_scalar, 
                               const Tensor& other, Tensor& out);

template<typename T>
void cuda_where_tensor_scalar(const Tensor& condition, const Tensor& input, 
                               T other_scalar, Tensor& out);

template<typename T, typename U>
void cuda_where_scalar_scalar(const Tensor& condition, T input_scalar, 
                               U other_scalar, Tensor& out);

// Public API - main where function
Tensor where(const Tensor& condition, const Tensor& input, const Tensor& other);

// Scalar overloads
template <typename T>
Tensor where(const Tensor& condition, T input_scalar, const Tensor& other);

template <typename T>
Tensor where(const Tensor& condition, const Tensor& input, T other_scalar);

// Two-template version handles both same-type AND mixed-type scalars
template <typename T, typename U>
Tensor where(const Tensor& condition, T input_scalar, U other_scalar);


} // namespace OwnTensor

#endif
