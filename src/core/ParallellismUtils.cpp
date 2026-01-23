#include "core/Tensor.h"
#include "core/TensorDispatch.h"
#include "device/DeviceTransfer.h"
#include "core/Views/ViewUtils.h"
#include <numeric>
#include <iostream>

namespace OwnTensor
{
    TensorOptions Tensor::opts()
    {
        TensorOptions opts;
        opts.dtype = this->impl_->dtype();
        opts.device = this->impl_->device();
        opts.requires_grad = this->impl_->requires_grad();

        return opts;
    }

    Tensor Tensor::slice(size_t start, size_t length)
    {
        TensorOptions opts = this->opts();


        if (start >= this->numel() || start + length >= this->numel())
        {
            throw std::runtime_error("Range exceeded!! (Zero based indexing)");
        }

        Tensor new_tensor = Tensor({ {1, static_cast<int64_t>(length)} }, opts);

        dispatch_by_dtype(this->dtype(), [&](auto dummy)
            {
                using T = decltype(dummy);

                void* temp_pointer = static_cast<T*>(this->data()) + start;

                device::copy_memory(new_tensor.data(), opts.device.device,
                    temp_pointer, opts.device.device,
                    length * dtype_size(this->dtype()));
            });

        return new_tensor;
    }

    Tensor Tensor::flatten_concat(std::vector<Tensor>& tensor_list)
    {
        // size_t total_elements = [&](const auto& list) -> size_t{
        //     size_t total = 0;
        //     for(auto& curr_tensor : tensor_list){ total += curr_tensor.numel(); }
        //     return total;
        // }(tensor_list);

        int64_t total_elements = std::accumulate(tensor_list.begin(), tensor_list.end(), int64_t(0),
            [](int64_t sum, const auto& tensor)
            {
                return sum + tensor.numel();
            }
        );

        Tensor result = Tensor({ {1, total_elements} }, tensor_list[0].opts());
        void* result_ptr = result.data();
        int64_t running_pointer = 0;



        for (const auto& tensor : tensor_list)
        {
            dispatch_by_dtype(tensor.dtype(), [&](auto dummy)
                {
                    using T = decltype(dummy);
                    void* new_ptr = (static_cast<T*>(result_ptr) + running_pointer);
                    device::copy_memory(new_ptr, result.device().device, tensor.data(), tensor.device().device, tensor.nbytes());
                    running_pointer += tensor.numel();

                });
        }
        return result;

    }

    std::vector<Tensor> Tensor::make_shards(size_t num_shards, bool row_major)
    {
        if (!row_major)
        {
            return this->t().contiguous().make_shards(num_shards, true);
        }

        if (this->numel() % num_shards != 0)
        {
            throw std::runtime_error("Cannot split this tensor into this number of equal shards");
        }

        TensorOptions opts = this->opts();

        size_t shard_elems = this->numel() / num_shards;
        size_t elem_bytes = dtype_size(this->dtype());
        size_t shard_bytes = shard_elems * elem_bytes;

        std::vector<Tensor> shards;
        shards.reserve(num_shards);

        uint8_t* base = static_cast<uint8_t*>(this->data());

        for (size_t i = 0; i < num_shards; ++i)
        {
            Tensor shard({ {1, static_cast<int64_t>(shard_elems)} }, opts);

            device::copy_memory(shard.data(), shard.device().device,
                base + i * shard_bytes, this->device().device, shard_bytes);

            shards.push_back(std::move(shard));
        }

        return shards;
    }

    std::vector<Tensor> Tensor::make_shards_cust(std::vector<Shape> shard_shapes, bool row_major)
    {

        if (!row_major)
        {
            return this->t().contiguous().make_shards_cust(shard_shapes, true);
        }

        TensorOptions opts = this->opts();
        int64_t total_req_elements = 0;


        for (const auto& s : shard_shapes)
        {
            int64_t shape_numel = 1;
            for (auto d : s.dims) shape_numel *= d;
            total_req_elements += shape_numel;
        }

        if (total_req_elements != static_cast<int64_t>(this->numel()))
        {
            throw std::runtime_error("make shards custom: Total elements in requested shapes ("
                + std::to_string(total_req_elements) + ") does not match the tensor suze ("
                + std::to_string(this->numel()) + ")");
        }

        size_t num_shards = shard_shapes.size();
        std::vector<Tensor> shards;
        shards.reserve(num_shards);


        uint8_t* base_ptr = static_cast<uint8_t*>(this->data());
        for (size_t i = 0; i < num_shards; ++i)
        {
            Tensor shard({ shard_shapes[i] }, opts);
            size_t elem_size = dtype_size(this->dtype());

            if (i == 0)
            {
                device::copy_memory(shard.data(), shard.device().device,
                    base_ptr, this->device().device,
                    elem_size * shard.numel());

                base_ptr += (shard.numel() * elem_size);
                shards.push_back(std::move(shard));
            }

            else
            {
                device::copy_memory(shard.data(), shard.device().device,
                    base_ptr, this->device().device,
                    elem_size * shard.numel());

                base_ptr += (shard.numel() * elem_size);
                shards.push_back(std::move(shard));
            }
        }

        return shards;
    }

    std::vector<Tensor> Tensor::make_shards_inplace(size_t num_shards, bool row_major)
    {
        if (!row_major)
        {
            return this->t().contiguous().make_shards_inplace(num_shards, true);
        }

        if (this->numel() % num_shards != 0)
        {
            throw std::runtime_error("Cannot split tensor into equal shards");
        }

        size_t shard_elems = this->numel() / num_shards;

        std::vector<Tensor> shards;
        shards.reserve(num_shards);

        for (size_t i = 0; i < num_shards; ++i)
        {
            size_t shard_offset_elems =
                this->storage_offset() + i * shard_elems;  // ELEMENT offset
            Shape shard_shape = Shape({ {1, (int64_t)shard_elems} });
            Tensor shard(                    // shared storage
                Shape(shard_shape),
                // ViewUtils::compute_strides(shard_shape),
                // static_cast<int64_t>(shard_offset_elems * dtype_size(this->dtype())),   // view offset
                this->dtype(),
                this->device(),
                false
            );

            shards.push_back(std::move(shard));
        }

        return shards;
    }

    std::vector<Tensor> Tensor::make_shards_inplace_cust(std::vector<Shape> shard_shapes, bool row_major)
    {
        if (!row_major)
        {
            return this->t().contiguous().make_shards_inplace_cust(shard_shapes, true);
        }
        int64_t total_req_elements = 0;
        for (const auto& s : shard_shapes)
        {
            int64_t shape_numel = 1;
            for (auto d : s.dims) shape_numel *= d;
            total_req_elements += shape_numel;
        }

        if (total_req_elements != static_cast<int64_t>(this->numel()))
        {
            throw std::runtime_error("make shards custom: Total elements in requested shapes ("
                + std::to_string(total_req_elements) + ") does not match the tensor suze ("
                + std::to_string(this->numel()) + ")");
        }


        std::vector<Tensor> shards;
        shards.reserve(shard_shapes.size());

        size_t shard_offset_elems = 0;

        {
    //     for (size_t i = 0; i < shard_shapes.size(); ++i)
    //     {


    //         // if (i == 0)
    //         // {
    //         //     Shape shard_shape = shard_shapes[i];

    //         //     Tensor shard(
    //         //         this->data_ptr_,                     // shared storage
    //         //         shard_shape,
    //         //         ViewUtils::compute_strides(shard_shape),
    //         //         0,   // view offset
    //         //         this->dtype_,
    //         //         this->device_,
    //         //         this->requires_grad_
    //         //     );

    //         //     shards.push_back(std::move(shard));
    //         // }
    //         // else
    //         // {

    //         Shape shard_shape = shard_shapes[i];

    //         Tensor shard(
    //             this->data_ptr_,                     // shared storage
    //             shard_shape,
    //             ViewUtils::compute_strides(shard_shape),
    //             (i == 0) ? 0 : (shard_offset_elems * dtype_size(this->dtype_)),   // view offset
    //             this->dtype_,
    //             this->device_,
    //             this->requires_grad_
    //         );

    //             shard_offset_elems += this->storage_offset_ + shards[i].numel();  // ELEMENT offset
    //             shards.push_back(std::move(shard));
    //         // }
    //     }
    //     return shards;
    // }
        }

        for (size_t i = 0; i < shard_shapes.size(); ++i)
        {
            Shape shard_shape = shard_shapes[i];

            // Calculate byte offset: (Current Element Offset * Bytes Per Element)
            size_t byte_offset = shard_offset_elems * dtype_size(this->dtype());

            Tensor shard(                   
                shard_shape,
                // ViewUtils::compute_strides(shard_shape),
                // byte_offset,   
                this->dtype(),
                this->device(),
                this->requires_grad()
            );

            // Increment the tracker by the number of elements in the shard just created
            shard_offset_elems += shard.numel();
            shards.push_back(std::move(shard));
        }
        return shards;
    }

}