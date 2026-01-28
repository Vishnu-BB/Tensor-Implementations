#include "device/AllocatorRegistry.h"
#include "device/CPUAllocator.h"
#include "device/CUDAAllocator.h"
#include "device/PinnedCPUAllocator.h"
#include "device/CachingCudaAllocator.h"

namespace OwnTensor
{ 
    namespace {
        CPUAllocator cpu_allocator;
        CUDAAllocator cuda_allocator;
        device::PinnedCPUAllocator pinned_cpu_allocator;   
    }

    Allocator* AllocatorRegistry::get_allocator(Device device) {
        if (device == Device::CPU) {
            return &cpu_allocator;
        } else if (device == Device::CUDA){
            return &CachingCUDAAllocator::instance();
        } else {
            return &cuda_allocator;
        }
    }

    Allocator* AllocatorRegistry::get_cpu_allocator() {
        return &cpu_allocator;
    }
    
    Allocator* AllocatorRegistry::get_pinned_cpu_allocator() {
        return &pinned_cpu_allocator;
    }

    Allocator* AllocatorRegistry::get_cuda_allocator() {
        return &cuda_allocator;
    }

    Allocator* AllocatorRegistry::get_caching_allocator()
    {
        return &CachingCUDAAllocator::instance();
    }

}