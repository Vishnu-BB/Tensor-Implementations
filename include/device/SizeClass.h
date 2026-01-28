#pragma once

#include <cstddef>

namespace OwnTensor
{
    class SizeClass {
        public:
            // rounding upto appropriate size grp
            static size_t round_size(size_t size) {
                if (size < kSmallSize) {
                    return ((size + 511) / 512) * 512;
                } else if (size < kSmallBuffer) {
                    return kSmallBuffer;
                } else if (size < kLargeBuffer) {
                    return kLargeBuffer;
                } else {
                    return ((size + kRoundLarge - 1) / kRoundLarge) * kRoundLarge;
                }
            }

            // to use small pool whenever possible
            static bool is_small(size_t size) {
                return size <= kSmallSize;
            }

            // Thresholds
            static constexpr size_t kSmallSize = 1048576;
            static constexpr size_t kSmallBuffer = 1048576 * 2;
            static constexpr size_t kLargeBuffer = 1048576 * 20;
            static constexpr size_t kMinLargeAlloc = 1048576 * 10;
            static constexpr size_t kRoundLarge = 2097152;
    };
}