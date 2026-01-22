#pragma once
#include "device/Allocator.h"
#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace OwnTensor {
namespace device {

class PinnedCPUAllocator : public Allocator {
public:
    void* allocate(size_t bytes) override;
    void deallocate(void* ptr) override;
    
    // For pinned memory, we can use standard memset/memcpy, 
    // or cudaMemset/Memcpy if we want to be explicit, but host pointers work with standard calls.
    // However, cudaMemcpyAsync from/to this memory is what matters.
    
    void memset(void* ptr, int value, size_t bytes) override;
    void memcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) override;
    
    void memsetAsync(void* ptr, int value, size_t bytes, cudaStream_t stream) override;
    void memcpyAsync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) override;
};

} // namespace device
} // namespace OwnTensor
