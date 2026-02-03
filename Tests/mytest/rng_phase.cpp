#include "core/Tensor.h"
#include "core/RNG.h"
#include <iostream>
#include <cassert>
#include <vector>

using namespace OwnTensor;

void test_cpu_rng_state() {
    std::cout << "Testing CPU RNG state capture/restore...\n";
    
    RNG::set_seed(425);
    
    // 1. Capture state BEFORE generating
    RNGState state = RNG::get_state();
    
    // 2. Generate first batch
    Tensor t1 = Tensor::rand<float>(Shape{{5}}, TensorOptions().with_device(Device::CPU));
    std::vector<float> v1(5);
    std::copy(t1.data<float>(), t1.data<float>() + 5, v1.begin());
    
    std::cout << "First batch: ";
    for(float i : v1) std::cout << i << " ";
    std::cout << "\n";
    
    // 3. Restore state
    RNG::set_state(state);
    
    // 4. Generate again - should match t1
    Tensor t2 = Tensor::rand<float>(Shape{{5}}, TensorOptions().with_device(Device::CPU));
    std::vector<float> v2(5);
    std::copy(t2.data<float>(), t2.data<float>() + 5, v2.begin());
    
    std::cout << "Second batch (after restore): ";
    for(float i : v2) std::cout << i << " ";
    std::cout << "\n";
    
    for (int i = 0; i < 5; ++i) {
        assert(v1[i] == v2[i]);
    }

    std::cout << "✓ CPU RNG state test passed!\n";
}

#ifdef WITH_CUDA
void test_gpu_rng_state() {
    std::cout << "\nTesting GPU RNG state capture/restore...\n";
    
    RNG::set_seed(455);
    
    // 1. Capture state BEFORE generating
    RNGState state = RNG::get_state();
    
    // 2. Generate first batch
    Tensor t1 = Tensor::rand<float>(Shape{{5}}, TensorOptions().with_device(Device::CUDA));
    std::vector<float> v1(5);
    cudaMemcpy(v1.data(), t1.data(), 5 * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "First batch: ";
    for(float i : v1) std::cout << i << " ";
    std::cout << "\n";
    
    // 3. Restore state
    RNG::set_state(state);
    
    // 4. Generate again - should match t1
    Tensor t2 = Tensor::rand<float>(Shape{{5}}, TensorOptions().with_device(Device::CUDA));
    std::vector<float> v2(5);
    cudaMemcpy(v2.data(), t2.data(), 5 * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "Second batch (after restore): ";
    for(float i : v2) std::cout << i << " ";
    std::cout << "\n";
    
    for (int i = 0; i < 5; ++i) {
        assert(v1[i] == v2[i]);
    }
    
    std::cout << "✓ GPU RNG state test passed!\n";
}
#endif

int main() {
    test_cpu_rng_state();
#ifdef WITH_CUDA
    test_gpu_rng_state();
#endif
    std::cout << "\nAll RNG tests passed!\n";
    return 0;
}