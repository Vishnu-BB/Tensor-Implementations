#pragma once

#include "device/Allocator.h"
#include "device/Block.h"
#include "device/BlockPool.h"
#include <vector>
#include <memory>
#include <deque>

namespace OwnTensor
{
    class CachingCUDAAllocator : public Allocator {
        public:
            static CachingCUDAAllocator& instance();

            void* allocate(size_t bytes) override;
            void deallocate(void* ptr) override;
            void memset(void* ptr, int value, size_t bytes) override; 
            void memcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) override;
            void memsetAsync(void* ptr, int value, size_t bytes, cudaStream_t stream) override; 
            void memcpyAsync(void* dst, const void* src, size_t bytes, 
                cudaMemcpyKind kind, cudaStream_t stream) override;

            
            void* allocate(size_t bytes, cudaStream_t stream);

            void empty_cache();

            void trim_to(size_t target_bytes);
            void trim_pool(BlockPool& pool, size_t target_bytes);

            struct MemoryStats {
                size_t allocated;
                size_t cached;
                size_t peak;
                size_t num_allocs;
                size_t num_frees;
                size_t num_cache_hits;
            };

            MemoryStats get_stats(int device = -1) const;

            void print_memory_summary() const;
            BlockPool& get_pool(size_t size, int device);

        private:
            CachingCUDAAllocator();
            ~CachingCUDAAllocator();

            struct DevicePools {
                BlockPool small_pool;
                BlockPool large_pool;
                std::unique_ptr<std::mutex> mtx; // Global lock for both pools on this device
                
                DevicePools() : mtx(std::make_unique<std::mutex>()) {}
                DevicePools(DevicePools&& other) noexcept = default;
                DevicePools& operator=(DevicePools&& other) noexcept = default;
            };

            std::deque<DevicePools> device_pools_;

            Block* cuda_alloc(size_t size, int device, cudaStream_t stream);

            void cuda_free(Block* block);
            void cuda_free_locked(Block* block);

            Block* try_split(Block* block, size_t size);

            void ensure_stream_safety(Block* block, cudaStream_t target_stream);

            mutable std::mutex stats_mutex_;
            size_t total_allocs_ = 0;
            size_t total_frees_ = 0;
            size_t cache_hits_ = 0;
    };
}