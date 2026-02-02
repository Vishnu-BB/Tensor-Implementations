#include "core/Tensor.h"
#include "core/TensorDispatch.h"
#include <random>
#include "device/DeviceCore.h"//✨✨✨



#include <cuda_runtime.h>
#include <curand.h>
#include "ops/helpers/ConversionKernels.cuh"
#include "core/RNG.h"

namespace OwnTensor
{
    // Helper for CUDA RNG
    void cuda_rand_uniform(float* data, size_t count, cudaStream_t stream)
    {
        curandGenerator_t gen = RNG::get_gpu_generator();
        curandSetStream(gen, stream);
        curandGenerateUniform(gen, data, count);
        RNG::increment_gpu_offset(count);
    }

    void cuda_rand_uniform(double* data, size_t count, cudaStream_t stream)
    {
        curandGenerator_t gen = RNG::get_gpu_generator();
        curandSetStream(gen, stream);
        curandGenerateUniformDouble(gen, data, count);
        RNG::increment_gpu_offset(count);
    }

    void cuda_rand_normal(float* data, size_t count, float sd, cudaStream_t stream)
    {
        curandGenerator_t gen = RNG::get_gpu_generator();
        curandSetStream(gen, stream);
        curandGenerateNormal(gen, data, count, 0.0f, float(sd));
        RNG::increment_gpu_offset(count);
    }

    void cuda_rand_normal(double* data, size_t count, double sd, cudaStream_t stream)
    {
        curandGenerator_t gen = RNG::get_gpu_generator();
        curandSetStream(gen, stream);
        curandGenerateNormalDouble(gen, data, count, 0.0, sd);
        RNG::increment_gpu_offset(count);
    }


    Tensor Tensor::zeros(Shape shape, TensorOptions opts)
    {
        Tensor tensor(shape, opts);

        if (opts.device.is_cpu())
        {
            // CPU implementation - handles all 7 types automatically
            dispatch_by_dtype(opts.dtype, [&](auto dummy)
                {
                    using T = decltype(dummy);
                    tensor.fill(T(0.0f));
                });
        }
        else
        {
            // GPU implementation - optimized with cudaMemset
#ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();//✨✨✨
            cudaMemsetAsync(tensor.data(), 0, tensor.nbytes(), stream);//✨✨✨
#else
            throw std::runtime_error("CUDA not available");
#endif
        }
        return tensor;
    }

    Tensor Tensor::empty(Shape shape, TensorOptions opts)
    {
        Tensor tensor(shape, opts);

        if (opts.device.is_cpu())
        {
            // CPU implementation - handles all 7 types automatically
            dispatch_by_dtype(opts.dtype, [&](auto dummy)
                {
                    // using T = decltype(dummy);
                    // tensor.fill(T(0.0f));
                });
        }
        else
        {
            // GPU implementation - optimized with cudaMemset
#ifdef WITH_CUDA
            // cudaStream_t stream = OwnTensor::cuda::getCurrentStream();//✨✨✨
            // cudaMemsetAsync(tensor.data(), 0, tensor.nbytes(), stream);//✨✨✨
#else
            throw std::runtime_error("CUDA not available");
#endif
        }
        return tensor;
    }

    Tensor Tensor::ones(Shape shape, TensorOptions opts)
    {
        Tensor tensor(shape, opts);

        if (opts.device.is_cpu())
        {
            // CPU implementation - handles all 7 types automatically
            // dispatch_by_dtype(opts.dtype, [&](auto dummy) {
            //     using T = decltype(dummy);
            //     tensor.fill(T(1.0f));
            // });
            dispatch_by_dtype(opts.dtype, [&](auto dummy)
                {
                    using T = decltype(dummy);
                    if constexpr (std::is_same_v<T, bool>)
                    {
                        // Special handling for bool: ones = true
                        tensor.fill(true);
                    }
                    else
                    {
                        tensor.fill(T(1.0f));
                    }
                });
        }
        else
        {
            // GPU implementation - handles all 7 types automatically
#ifdef WITH_CUDA
            [[maybe_unused]] cudaStream_t stream = OwnTensor::cuda::getCurrentStream();//✨✨✨
            dispatch_by_dtype(opts.dtype, [&](auto dummy)
                {
                    using T = decltype(dummy);
                    if constexpr (std::is_same_v<T, bool>)
                    {
                        // For bool on GPU, use memset with 1
                        cudaMemset(tensor.data(), 1, tensor.numel());
                    }
                    else
                    {
                        std::vector<T> ones_data(tensor.numel(), T(1.0f));
                        cudaMemcpy(tensor.data(), ones_data.data(),
                            tensor.numel() * sizeof(T), cudaMemcpyHostToDevice);
                    }
                });
#else
            throw std::runtime_error("CUDA not available");
#endif
        }
        return tensor;
    }

    Tensor Tensor::full(Shape shape, TensorOptions opts, float value)
    {
        Tensor tensor(shape, opts);

        if (opts.device.is_cpu())
        {
            dispatch_by_dtype(opts.dtype, [&](auto dummy)
                {
                    using T = decltype(dummy);
                    if constexpr (std::is_same_v<T, bool>)
                    {
                        // For bool: any nonzero value = true
                        tensor.fill(value != 0.0f);
                    }
                    else
                    {
                        tensor.fill(static_cast<T>(value));
                    }
                });
        }
        else
        {
#ifdef WITH_CUDA
            [[maybe_unused]] cudaStream_t stream = OwnTensor::cuda::getCurrentStream();//✨✨✨
            dispatch_by_dtype(opts.dtype, [&](auto dummy)
                {
                    using T = decltype(dummy);
                    if constexpr (std::is_same_v<T, bool>)
                    {
                        uint8_t bool_val = (value != 0.0f) ? 1 : 0;
                        cudaMemset(tensor.data(), bool_val, tensor.numel());
                    }
                    else
                    {
                        std::vector<T> fill_data(tensor.numel(), static_cast<T>(value));
                        cudaMemcpy(tensor.data(), fill_data.data(),
                            tensor.numel() * sizeof(T), cudaMemcpyHostToDevice);
                    }
                });
#else
            throw std::runtime_error("CUDA not available");
#endif
        }

        return tensor;
    }

    template <typename U>
    Tensor Tensor::rand(Shape shape, TensorOptions opts,unsigned long seed, U lower, U upper)
    {
        Tensor tensor(shape, opts);

        if (opts.device.is_cpu())
        {
            // CPU random
            if (seed != 0) {
                RNG::set_seed(seed);
            }
            auto& gen = RNG::get_cpu_generator();

            dispatch_by_dtype(opts.dtype, [&](auto dummy)
                {
                    using T = decltype(dummy);
                    if constexpr (std::is_floating_point_v<T>)
                    {
                        std::uniform_real_distribution<T> dist(lower, upper);
                        T* data = static_cast<T*>(tensor.data());
                        for (size_t i = 0; i < tensor.numel(); ++i)
                        {
                            data[i] = dist(gen);
                        }
                    }
                    else if constexpr (std::is_same_v<T, OwnTensor::float16_t> || std::is_same_v<T, OwnTensor::bfloat16_t>)
                    {
                        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                        T* data = static_cast<T*>(tensor.data());
                        for (size_t i = 0; i < tensor.numel(); ++i)
                        {
                            data[i] = static_cast<T>(dist(gen));
                        }
                    }
                    else
                    {
                        throw std::runtime_error("rand only supports floating point types");
                    }
                });
        }
        else
        {
            // GPU random
#ifdef WITH_CUDA
            if (seed != 0) {
                RNG::set_seed(seed);
            }
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();

            dispatch_by_dtype(opts.dtype, [&](auto dummy)
                {
                    using T = decltype(dummy);
                    if constexpr (std::is_same_v<T, float>)
                    {
                        cuda_rand_uniform(static_cast<float*>(tensor.data()), tensor.numel(), stream);
                    }
                    else if constexpr (std::is_same_v<T, double>)
                    {
                        cuda_rand_uniform(static_cast<double*>(tensor.data()), tensor.numel(), stream);
                    }
                    else if constexpr (std::is_same_v<T, OwnTensor::float16_t> || std::is_same_v<T, OwnTensor::bfloat16_t>)
                    {
                        // 1. Allocate temporary float buffer on GPU
                        float* temp_data;
                        cudaMallocAsync(&temp_data, tensor.numel() * sizeof(float), stream);
                        cuda_rand_uniform(temp_data, tensor.numel(), stream);
                        convert_type_cuda(temp_data, static_cast<T*>(tensor.data()), tensor.numel(), stream);
                        cudaFreeAsync(temp_data, stream);
                    }
                    else
                    {
                        throw std::runtime_error("GPU rand only supports float/double/half/bfloat16");
                    }
                });
#else
            throw std::runtime_error("CUDA not available");
#endif
        }

        return tensor;
    }

    template <typename U>
    Tensor Tensor::randn(Shape shape, TensorOptions opts,unsigned long seed , U sd)
    {
        Tensor tensor(shape, opts);

        if (opts.device.is_cpu())
        {
            // CPU random
            if (seed != 0) {
                RNG::set_seed(seed);
            }
            auto& gen = RNG::get_cpu_generator();

            dispatch_by_dtype(opts.dtype, [&](auto dummy)
                {
                    using T = decltype(dummy);
                    if constexpr (std::is_floating_point_v<T>)
                    {
                        std::normal_distribution<T> dist(0.0, sd);
                        T* data = static_cast<T*>(tensor.data());
                        for (size_t i = 0; i < tensor.numel(); ++i)
                        {
                            data[i] = dist(gen);
                        }
                    }
                    else if constexpr (std::is_same_v<T, OwnTensor::float16_t> || std::is_same_v<T, OwnTensor::bfloat16_t>)
                    {
                        std::normal_distribution<float> dist(0.0f, float(sd));
                        T* data = static_cast<T*>(tensor.data());
                        for (size_t i = 0; i < tensor.numel(); ++i)
                        {
                            data[i] = static_cast<T>(dist(gen));
                        }
                    }
                    else
                    {
                        throw std::runtime_error("randn only supports floating point types");
                    }
                });
        }
        else
        {
            // GPU random
#ifdef WITH_CUDA
            if (seed != 0) {
                RNG::set_seed(seed);
            }
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();

            dispatch_by_dtype(opts.dtype, [&](auto dummy)
                {
                    using T = decltype(dummy);
                    if constexpr (std::is_same_v<T, float>)
                    {
                        cuda_rand_normal(static_cast<float*>(tensor.data()), tensor.numel(), sd, stream);
                    }
                    else if constexpr (std::is_same_v<T, double>)
                    {
                        cuda_rand_normal(static_cast<double*>(tensor.data()), tensor.numel(), sd, stream);
                    }
                    else if constexpr (std::is_same_v<T, OwnTensor::float16_t> || std::is_same_v<T, OwnTensor::bfloat16_t>)
                    {
                        // 1. Allocate temporary float buffer on GPU
                        float* temp_data;
                        cudaMallocAsync(&temp_data, tensor.numel() * sizeof(float), stream);
                        cuda_rand_normal(temp_data, tensor.numel(), float(sd), stream);
                        convert_type_cuda(temp_data, static_cast<T*>(tensor.data()), tensor.numel(), stream);
                        cudaFreeAsync(temp_data, stream);
                    }
                    else
                    {
                        throw std::runtime_error("GPU randn only supports float/double");
                    }
                });
#else
            throw std::runtime_error("CUDA not available");
#endif
        }

        return tensor;
    }

    template Tensor Tensor::rand<float>(Shape shape, TensorOptions opts,unsigned long seed, float lower, float upper);
    template Tensor Tensor::rand<double>(Shape shape, TensorOptions opts,unsigned long seed, double lower, double upper);

    template Tensor Tensor::randn<float>(Shape shape, TensorOptions opts,unsigned long seed, float sd);
    template Tensor Tensor::randn<double>(Shape shape, TensorOptions opts,unsigned long seed, double sd);

}