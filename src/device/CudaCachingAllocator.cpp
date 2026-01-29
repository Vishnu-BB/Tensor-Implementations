#include "device/CachingCudaAllocator.h"
#include "device/SizeClass.h"
#include "device/AllocationTracker.h"
#include "device/DeviceCore.h"
#include <iostream>
#include <algorithm>

namespace OwnTensor
{
    CachingCUDAAllocator& CachingCUDAAllocator::instance()
    {
        static CachingCUDAAllocator allocator;
        return allocator;
    }

    CachingCUDAAllocator::CachingCUDAAllocator()
    {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        device_pools_.resize(device_count);
    }

    CachingCUDAAllocator::~CachingCUDAAllocator()
    {
        empty_cache();
    }

    void* CachingCUDAAllocator::allocate(size_t bytes)
    {
        cudaStream_t stream = cuda::getCurrentStream();
        return allocate(bytes, stream);
    }

    void CachingCUDAAllocator::trim_to(size_t target_bytes)
    {
        for (DevicePools& pools : device_pools_)
        {
            std::lock_guard<std::mutex> lock(*pools.mtx);
            trim_pool(pools.small_pool, target_bytes / 2);
            trim_pool(pools.large_pool, target_bytes / 2);
        }
    }

    void CachingCUDAAllocator::trim_pool(BlockPool& pool, size_t target_bytes) {
    // ASSUMES LOCK IS HELD

    // Use a while loop with a safe forward-iterator check
    while (pool.total_cached > target_bytes && !pool.free_blocks.empty()) {
        
        // Always take the largest block to satisfy OOM quickly
        // In std::set, the largest element is at the end
        auto it = std::prev(pool.free_blocks.end());
        Block* block = *it;

        // CRITICAL: Update pointers and stats BEFORE the slow cudaFree
        pool.total_cached -= block->size;
        // pool.allocated_blocks.erase(block->ptr); // Not in allocated blocks if it's in free_blocks
        pool.free_blocks.erase(it);

        // LOCK #2: Stats update (must be very brief)
        {
            std::lock_guard<std::mutex> s_lock(stats_mutex_);
            total_frees_++;
        }

        // PHYSICAL FREE
        // We do this while still holding pool.mtx to prevent 
        // another thread from getting the same pointer from the driver
        cudaStreamSynchronize(block->stream);
        cudaFree(block->ptr); 
        
        delete block;
    }
}



    void* CachingCUDAAllocator::allocate(size_t bytes, cudaStream_t stream)
    {
        if (bytes == 0) return nullptr;

        int device;
        cudaGetDevice(&device);

        size_t alloc_size = SizeClass::round_size(bytes);

        BlockPool& pool = get_pool(alloc_size, device);

        Block* block = nullptr;
        {
            std::lock_guard<std::mutex> lock(*device_pools_[device].mtx);
            block = pool.find_free_block(alloc_size, stream);

            if (block)
            {
                cache_hits_ += 1;
                pool.total_cached -= block->size;

                ensure_stream_safety(block, stream);

                if (block->size >= alloc_size + SizeClass::kSmallSize)
                {
                    block = try_split(block, alloc_size);
                }
            }
        }

        // Inside allocate()
if (!block) {
    block = cuda_alloc(alloc_size, device, stream);
    if (!block) {
        // Log to system stderr directly to bypass potential stream hangs
        fprintf(stderr, "[ALLOCATOR] Hard OOM. Trimming cache...\n");
        
        this->trim_to(0); 
        
        block = cuda_alloc(alloc_size, device, stream);
        if (!block) {
             fprintf(stderr, "[ALLOCATOR] FATAL: Still OOM after trim.\n");
             throw std::runtime_error("CUDA OOM");
        }
    }
}


        block->allocated = true;
        block->req_size = bytes;
        block->stream = stream;

        {
            std::lock_guard<std::mutex> lock(*device_pools_[device].mtx);
            pool.allocated_blocks[block->ptr] = block;
        }

        // AllocationTracker::instance().on_alloc(block->ptr, bytes, device);

        total_allocs_++;
        return block->ptr;
    }

    void CachingCUDAAllocator::deallocate(void* ptr)
    {
        if (!ptr) return;

        int device;
        cudaGetDevice(&device);

        Block* block = nullptr;
        BlockPool* pool_ptr = nullptr;

        {
            std::lock_guard<std::mutex> lock(*device_pools_[device].mtx);
            for (BlockPool* pool :
                { &device_pools_[device].small_pool, &device_pools_[device].large_pool })
            {
                auto it = pool->allocated_blocks.find(ptr);
                if (it != pool->allocated_blocks.end())
                {
                    block = it->second;
                    pool_ptr = pool;
                    pool->allocated_blocks.erase(it);
                    break;
                }
            }

            if (!block)
            {
                std::cerr << "Warning: deallocate called on unknown pointer" << std::endl;
                return;
            }

            // AllocationTracker::instance().on_free(ptr, device);

            block->allocated = false;
            pool_ptr->try_coalesce(block);
            pool_ptr->free_blocks.insert(block);
            pool_ptr->total_cached += block->size;
        }

        total_frees_++;
    }

    Block* CachingCUDAAllocator::cuda_alloc(size_t size, int device, cudaStream_t stream)
    {
        void* ptr = nullptr;
        cudaError_t err = cudaMallocAsync(&ptr, size, stream);
        if (err != cudaSuccess || !ptr) {
            fprintf(stderr, "[ALLOCATOR] cudaMallocAsync(size=%zu) failed: %s\n", size, cudaGetErrorString(err));
            return nullptr;
        }

        Block* block = new Block(ptr, size, device, stream);

        BlockPool& pool = get_pool(size, device);
        {
            std::lock_guard<std::mutex> lock(*device_pools_[device].mtx);
            pool.total_allocated += size;
            pool.peak_allocated = std::max(pool.peak_allocated, pool.total_allocated);
        }
        return block;
    }


    void CachingCUDAAllocator::cuda_free(Block* block)
    {
        cudaFreeAsync(block->ptr, block->stream);
        BlockPool& pool = get_pool(block->size, block->device_id);
        {
            std::lock_guard<std::mutex> lock(*device_pools_[block->device_id].mtx);
            pool.total_allocated -= block->size;
        }

        delete block;
    }

    void CachingCUDAAllocator::cuda_free_locked(Block* block)
    {
        // ASSUMES LOCK IS HELD
        cudaFreeAsync(block->ptr, block->stream);
        BlockPool& pool = get_pool(block->size, block->device_id);
        pool.total_allocated -= block->size;
        delete block;
    }

    BlockPool& CachingCUDAAllocator::get_pool(size_t size, int device)
    {
        if (SizeClass::is_small(size))
        {
            return device_pools_[device].small_pool;
        }
        else
        {
            return device_pools_[device].large_pool;
        }
    }

    void CachingCUDAAllocator::empty_cache()
    {
        std::vector<Block*> blocks_to_free;
        for (DevicePools& dev_pools : device_pools_)
        {
             // Physical layout and pools are protected by device lock
             std::lock_guard<std::mutex> lock(*dev_pools.mtx);
             for (BlockPool* pool : { &dev_pools.small_pool, &dev_pools.large_pool })
             {
                for (Block* block : pool->free_blocks)
                {
                    blocks_to_free.push_back(block);
                }
                pool->free_blocks.clear();
                pool->total_cached = 0;
             }
        }

        for (Block* block : blocks_to_free)
        {
            // we already cleared the pools' total_cached and free_blocks lists
            // but we need to update total_allocated and call Physical Free
            cuda_free_locked(block);
        }
        
        cudaDeviceSynchronize();
    }

    void CachingCUDAAllocator::ensure_stream_safety(Block* block, cudaStream_t target_stream)
    {
        if (block->stream == target_stream)
        {
            return;
        }

        if (block->stream == 0 || target_stream == 0)
        {
            cudaStreamSynchronize(block->stream);
        }
        else
        {
            cudaEvent_t event;
            cudaEventCreate(&event);
            cudaEventRecord(event, block->stream);
            cudaStreamWaitEvent(target_stream, event, 0);
            cudaEventDestroy(event);
        }

        block->stream = target_stream;
    }

    Block* CachingCUDAAllocator::try_split(Block* block, size_t size)
    {
        size_t remaining = block->size - size;

        if (remaining < SizeClass::kSmallSize)
        {
            return block; // too small to split }
        }
        void* new_ptr = static_cast<char*>(block->ptr) + size;
        Block* new_block = new Block(new_ptr, remaining, block->device_id, block->stream);
        new_block->is_split = true;

        block->size = size;
        block->is_split = true;

        new_block->prev = block;
        new_block->next = block->next;
        if (block->next)
        {
            block->next->prev = new_block;
        }
        block->next = new_block;


        BlockPool& pool = get_pool(remaining, block->device_id);
        // Already holding device lock from allocate()
        pool.free_blocks.insert(new_block);
        pool.total_cached += remaining;
        
        return block;
    }

    Block* BlockPool::find_free_block(size_t size, cudaStream_t /*stream*/)
    {
        Block search_key(nullptr, size, 0, nullptr);
        auto it = free_blocks.lower_bound(&search_key);

        if (it != free_blocks.end())
        {
            Block* block = *it;
            free_blocks.erase(it);
            return block;
        }
        return nullptr;
    }

    void BlockPool::return_block(Block* block)
    {
        free_blocks.insert(block);
        total_cached += block->size; 
    }

    void BlockPool::try_coalesce(Block* block)
    {
        // This is called from CachingCUDAAllocator with the device lock held.
        // We need to be careful because neighbors might be in different pools.
        
        CachingCUDAAllocator& alloc = CachingCUDAAllocator::instance();

        if (block->prev && !block->prev->allocated)
        {
            Block* prev = block->prev;
            BlockPool& prev_pool = alloc.get_pool(prev->size, prev->device_id);
            
            prev_pool.free_blocks.erase(prev);
            prev_pool.total_cached -= prev->size;
            
            prev->size += block->size;
            prev->next = block->next;
            if (block->next)
            {
                block->next->prev = prev;
            }

            delete block;
            block = prev;
        }

        if (block->next && !block->next->allocated)
        {
            Block* next = block->next;
            BlockPool& next_pool = alloc.get_pool(next->size, next->device_id);
            
            next_pool.free_blocks.erase(next);
            next_pool.total_cached -= next->size;
            
            block->size += next->size;
            block->next = next->next;
            if (next->next)
            {
                next->next->prev = block;
            }

            delete next;
        }
    }

    CachingCUDAAllocator::MemoryStats CachingCUDAAllocator::get_stats(int device) const
    {
        MemoryStats stats = {};

        auto add_pool_stats = [&](const BlockPool& pool)
            {
                stats.allocated += pool.total_allocated - pool.total_cached;
                stats.cached += pool.total_cached;
                stats.peak = std::max(stats.peak, pool.peak_allocated);
            };

        if (device < 0)
        {
            for (const DevicePools& dev_pools : device_pools_)
            {
                std::lock_guard<std::mutex> lock(*dev_pools.mtx);
                add_pool_stats(dev_pools.small_pool);
                add_pool_stats(dev_pools.large_pool);
            }
        }
        else
        {
            if (device < (int)device_pools_.size()) {
                const DevicePools& dev_pools = device_pools_[device];
                std::lock_guard<std::mutex> lock(*dev_pools.mtx);
                add_pool_stats(dev_pools.small_pool);
                add_pool_stats(dev_pools.large_pool);
            }
        }

        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats.num_allocs = total_allocs_;
            stats.num_frees = total_frees_;
            stats.num_cache_hits = cache_hits_;
        }
        return stats;
    }

    void CachingCUDAAllocator::print_memory_summary() const
    {
        MemoryStats stats = get_stats();

        std::cerr << "\n========== CUDA Caching Allocator Stats ==========\n";
        std::cerr << "Allocated:    " << (stats.allocated / 1024.0 / 1024.0) << " MB\n";
        std::cerr << "Cached:       " << (stats.cached / 1024.0 / 1024.0) << " MB\n";
        std::cerr << "Peak:         " << (stats.peak / 1024.0 / 1024.0) << " MB\n";
        std::cerr << "Allocations:  " << stats.num_allocs << "\n";
        std::cerr << "Frees:        " << stats.num_frees << "\n";
        std::cerr << "Cache hits:   " << stats.num_cache_hits
            << " (" << (100.0 * stats.num_cache_hits / std::max(1UL, stats.num_allocs))
            << "%)\n";
        std::cerr << "===================================================\n\n";
    }

    void CachingCUDAAllocator::memset(void* ptr, int value, size_t bytes)
    {
        cudaStream_t stream = cuda::getCurrentStream();
        memsetAsync(ptr, value, bytes, stream);
    }

    void CachingCUDAAllocator::memcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind)
    {
        cudaStream_t stream = cuda::getCurrentStream();
        memcpyAsync(dst, src, bytes, kind, stream);
    }

    void CachingCUDAAllocator::memsetAsync(void* ptr, int value, size_t bytes, cudaStream_t stream)
    {
        cudaMemsetAsync(ptr, value, bytes, stream);
    }


    void CachingCUDAAllocator::memcpyAsync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream)
    {
        cudaMemcpyAsync(dst, src, bytes, kind, stream);
    }

}

