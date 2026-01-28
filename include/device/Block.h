#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace OwnTensor
{
    struct Block
    {
        // Memory Info
        void* ptr;
        size_t size;
        size_t req_size;

        // Pool
        int device_id;
        cudaStream_t stream;

        // for block splitting
        Block* prev;
        Block* next;

        // state of the block
        bool allocated;
        bool is_split;

        uint64_t alloc_id;

        Block(void* p, size_t s, int dev, cudaStream_t str)
            : ptr(p), size(s), req_size(s), device_id(dev), stream(str),
            prev(nullptr), next(nullptr), allocated(false), is_split(false),
            alloc_id(0)
        { }

    };

    // for requesting block by size
    struct BlockSizeComparator
    {
        bool operator() (const Block* a, const Block* b) const
        {
            if (a->size != b->size) return a->size < b->size;
            return a->ptr < b->ptr;
        }
    };

    // for requesting block by pointer
    struct BlockPtrComparator
    {
        bool operator() (const Block* a, const Block* b) const
        {
            return a->ptr < b->ptr;
        }
    };
}