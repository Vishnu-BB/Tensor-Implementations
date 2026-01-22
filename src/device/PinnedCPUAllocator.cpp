#include "device/PinnedCPUAllocator.h"
#include <cstring>
#include <stdexcept>
#include <iostream>

namespace OwnTensor {
namespace device {

void* PinnedCPUAllocator::allocate(size_t bytes) {
#ifdef WITH_CUDA
    void* ptr = nullptr;
    // cudaHostAllocDefault: standard pinned memory
    cudaError_t err = cudaMallocHost(&ptr, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA Pinned Memory allocation failed.");
    }
    return ptr;
#else
    return new uint8_t[bytes]; // Fallback to standard allocation
#endif
}

void PinnedCPUAllocator::deallocate(void* ptr) {
#ifdef WITH_CUDA
    cudaFreeHost(ptr);
#else
    delete[] static_cast<uint8_t*>(ptr);
#endif
}

void PinnedCPUAllocator::memset(void* ptr, int value, size_t bytes) {
    std::memset(ptr, value, bytes);
}

void PinnedCPUAllocator::memcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
    (void)kind;
    std::memcpy(dst, src, bytes);
}

void PinnedCPUAllocator::memsetAsync(void* ptr, int value, size_t bytes, cudaStream_t stream) {
    (void)stream;
    // CPU operations are synchronous relative to CPU thread.
    // We could launch a kernel to memset host memory? No.
    std::memset(ptr, value, bytes); 
}

void PinnedCPUAllocator::memcpyAsync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
    (void)stream; (void)kind;
    // PinnedAllocator is for Host memory.
    // If we are copying Host->Host, use memcpy.
    // If copying Host->Device, this allocator is used for the HOST side via DeviceTransfer.
    // This method implementation assumes simple CPU copy for now.
    std::memcpy(dst, src, bytes);
}

} // namespace device
} // namespace OwnTensor
