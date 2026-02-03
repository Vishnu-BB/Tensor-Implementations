#include "core/Tensor.h"
#include "core/RNG.h"
#include "device/Device.h"
#include "dtype/Dtype.h"

#include <iostream>
#include <cassert>
#include <vector>
#include <iomanip>
#include <cmath>

using namespace OwnTensor;

// Helper function to print tensor values
template<typename T>
void print_values(const std::vector<T>& values, const std::string& label) {
    std::cout << label << ": ";
    for (const auto& val : values) {
        std::cout << std::fixed << std::setprecision(6) << val << " ";
    }
    std::cout << "\n";
}

// Helper function to compare float vectors with tolerance
bool compare_vectors(const std::vector<float>& v1, const std::vector<float>& v2, float epsilon = 1e-6) {
    if (v1.size() != v2.size()) return false;
    for (size_t i = 0; i < v1.size(); ++i) {
        if (std::abs(v1[i] - v2[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

// ==============================================================================
// TEST 1: Basic CPU RNG State Capture and Restore
// ==============================================================================
void test_cpu_rng_basic_state() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST 1: Basic CPU RNG State Capture and Restore\n";
    std::cout << std::string(80, '=') << "\n";

    RNG::set_seed(42);
    
    // Capture state before generation
    RNGState state1 = RNG::get_state();
    
    // Generate first tensor
    Tensor t1 = Tensor::rand<float>(Shape{{10}}, TensorOptions().with_device(Device::CPU));
    std::vector<float> v1(10);
    std::copy(t1.data<float>(), t1.data<float>() + 10, v1.begin());
    print_values(v1, "First generation");
    
    // Generate second tensor (state has advanced)
    Tensor t2 = Tensor::rand<float>(Shape{{10}}, TensorOptions().with_device(Device::CPU));
    std::vector<float> v2(10);
    std::copy(t2.data<float>(), t2.data<float>() + 10, v2.begin());
    print_values(v2, "Second generation (advanced state)");
    
    // Restore state to state1 and regenerate
    RNG::set_state(state1);
    Tensor t3 = Tensor::rand<float>(Shape{{10}}, TensorOptions().with_device(Device::CPU));
    std::vector<float> v3(10);
    std::copy(t3.data<float>(), t3.data<float>() + 10, v3.begin());
    print_values(v3, "Third generation (restored state)");
    
    // Verify t1 == t3
    assert(compare_vectors(v1, v3) && "Restored state should reproduce same values");
    assert(!compare_vectors(v1, v2) && "Advanced state should produce different values");
    
    std::cout << "✓ TEST 1 PASSED: CPU state restore works correctly\n";
}

// ==============================================================================
// TEST 2: CPU RNG Multiple State Snapshots
// ==============================================================================
void test_cpu_rng_multiple_snapshots() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST 2: CPU RNG Multiple State Snapshots\n";
    std::cout << std::string(80, '=') << "\n";

    RNG::set_seed(123);
    
    std::vector<RNGState> snapshots;
    std::vector<std::vector<float>> results;
    
    // Take 5 snapshots and generate tensors
    for (int i = 0; i < 5; i++) {
        snapshots.push_back(RNG::get_state());
        
        Tensor t = Tensor::rand<float>(Shape{{5}}, TensorOptions().with_device(Device::CPU));
        std::vector<float> v(5);
        std::copy(t.data<float>(), t.data<float>() + 5, v.begin());
        results.push_back(v);
        
        std::cout << "Snapshot " << i << " generated: ";
        print_values(v, "");
    }
    
    // Now restore each snapshot and verify we get same results
    std::cout << "\nVerifying snapshot restoration:\n";
    for (int i = 0; i < 5; i++) {
        RNG::set_state(snapshots[i]);
        Tensor t = Tensor::rand<float>(Shape{{5}}, TensorOptions().with_device(Device::CPU));
        std::vector<float> v(5);
        std::copy(t.data<float>(), t.data<float>() + 5, v.begin());
        
        std::cout << "Snapshot " << i << " restored:  ";
        print_values(v, "");
        
        assert(compare_vectors(results[i], v) && "Snapshot restoration failed");
    }
    
    std::cout << "✓ TEST 2 PASSED: Multiple snapshots work correctly\n";
}

// ==============================================================================
// TEST 3: CPU RNG - randn (Normal Distribution)
// ==============================================================================
void test_cpu_rng_normal_distribution() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST 3: CPU RNG - Normal Distribution State\n";
    std::cout << std::string(80, '=') << "\n";

    RNG::set_seed(789);
    
    // Capture state
    RNGState state = RNG::get_state();
    
    // Generate normal distribution
    Tensor t1 = Tensor::randn<float>(Shape{{15}}, TensorOptions().with_device(Device::CPU), 0, 2.0f);
    std::vector<float> v1(15);
    std::copy(t1.data<float>(), t1.data<float>() + 15, v1.begin());
    print_values(v1, "Normal dist (σ=2.0)");
    
    // Restore and regenerate
    RNG::set_state(state);
    Tensor t2 = Tensor::randn<float>(Shape{{15}}, TensorOptions().with_device(Device::CPU), 0, 2.0f);
    std::vector<float> v2(15);
    std::copy(t2.data<float>(), t2.data<float>() + 15, v2.begin());
    print_values(v2, "Normal dist (restored)");
    
    assert(compare_vectors(v1, v2) && "Normal distribution state restoration failed");
    
    std::cout << "✓ TEST 3 PASSED: Normal distribution state works correctly\n";
}

// ==============================================================================
// TEST 4: CPU RNG - Seeding Effects
// ==============================================================================
void test_cpu_rng_seeding() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST 4: CPU RNG - Seeding Effects\n";
    std::cout << std::string(80, '=') << "\n";

    // Same seed should produce same results
    RNG::set_seed(999);
    Tensor t1 = Tensor::rand<float>(Shape{{8}}, TensorOptions().with_device(Device::CPU));
    std::vector<float> v1(8);
    std::copy(t1.data<float>(), t1.data<float>() + 8, v1.begin());
    print_values(v1, "Seed 999 - first run");
    
    RNG::set_seed(999);
    Tensor t2 = Tensor::rand<float>(Shape{{8}}, TensorOptions().with_device(Device::CPU));
    std::vector<float> v2(8);
    std::copy(t2.data<float>(), t2.data<float>() + 8, v2.begin());
    print_values(v2, "Seed 999 - second run");
    
    assert(compare_vectors(v1, v2) && "Same seed should produce same results");
    
    // Different seed should produce different results
    RNG::set_seed(1000);
    Tensor t3 = Tensor::rand<float>(Shape{{8}}, TensorOptions().with_device(Device::CPU));
    std::vector<float> v3(8);
    std::copy(t3.data<float>(), t3.data<float>() + 8, v3.begin());
    print_values(v3, "Seed 1000");
    
    assert(!compare_vectors(v1, v3) && "Different seed should produce different results");
    
    std::cout << "✓ TEST 4 PASSED: Seeding works correctly\n";
}

// ==============================================================================
// TEST 5: CPU RNG - Mixed rand and randn
// ==============================================================================
void test_cpu_rng_mixed_operations() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST 5: CPU RNG - Mixed rand and randn Operations\n";
    std::cout << std::string(80, '=') << "\n";

    RNG::set_seed(555);
    
    // Capture state
    RNGState state = RNG::get_state();
    
    // Generate sequence: rand -> randn -> rand
    Tensor t1_uniform = Tensor::rand<float>(Shape{{5}}, TensorOptions().with_device(Device::CPU));
    Tensor t1_normal = Tensor::randn<float>(Shape{{5}}, TensorOptions().with_device(Device::CPU));
    Tensor t1_uniform2 = Tensor::rand<float>(Shape{{5}}, TensorOptions().with_device(Device::CPU));
    
    std::vector<float> v1_u1(5), v1_n(5), v1_u2(5);
    std::copy(t1_uniform.data<float>(), t1_uniform.data<float>() + 5, v1_u1.begin());
    std::copy(t1_normal.data<float>(), t1_normal.data<float>() + 5, v1_n.begin());
    std::copy(t1_uniform2.data<float>(), t1_uniform2.data<float>() + 5, v1_u2.begin());
    
    print_values(v1_u1, "Sequence 1: Uniform");
    print_values(v1_n, "Sequence 1: Normal");
    print_values(v1_u2, "Sequence 1: Uniform again");
    
    // Restore state and repeat
    RNG::set_state(state);
    
    Tensor t2_uniform = Tensor::rand<float>(Shape{{5}}, TensorOptions().with_device(Device::CPU));
    Tensor t2_normal = Tensor::randn<float>(Shape{{5}}, TensorOptions().with_device(Device::CPU));
    Tensor t2_uniform2 = Tensor::rand<float>(Shape{{5}}, TensorOptions().with_device(Device::CPU));
    
    std::vector<float> v2_u1(5), v2_n(5), v2_u2(5);
    std::copy(t2_uniform.data<float>(), t2_uniform.data<float>() + 5, v2_u1.begin());
    std::copy(t2_normal.data<float>(), t2_normal.data<float>() + 5, v2_n.begin());
    std::copy(t2_uniform2.data<float>(), t2_uniform2.data<float>() + 5, v2_u2.begin());
    
    print_values(v2_u1, "Sequence 2: Uniform");
    print_values(v2_n, "Sequence 2: Normal");
    print_values(v2_u2, "Sequence 2: Uniform again");
    
    assert(compare_vectors(v1_u1, v2_u1) && "First uniform failed");
    assert(compare_vectors(v1_n, v2_n) && "Normal failed");
    assert(compare_vectors(v1_u2, v2_u2) && "Second uniform failed");
    
    std::cout << "✓ TEST 5 PASSED: Mixed operations state works correctly\n";
}

#ifdef WITH_CUDA
// ==============================================================================
// TEST 6: GPU RNG Basic State Capture and Restore
// ==============================================================================
void test_gpu_rng_basic_state() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST 6: GPU RNG Basic State Capture and Restore\n";
    std::cout << std::string(80, '=') << "\n";

    RNG::set_seed(42);
    
    // Capture state before generation
    RNGState state1 = RNG::get_state();
    
    // Generate first tensor on GPU
    Tensor t1 = Tensor::rand<float>(Shape{{10}}, TensorOptions().with_device(Device::CUDA));
    std::vector<float> v1(10);
    cudaMemcpy(v1.data(), t1.data(), 10 * sizeof(float), cudaMemcpyDeviceToHost);
    print_values(v1, "GPU: First generation");
    
    // Generate second tensor (state has advanced)
    Tensor t2 = Tensor::rand<float>(Shape{{10}}, TensorOptions().with_device(Device::CUDA));
    std::vector<float> v2(10);
    cudaMemcpy(v2.data(), t2.data(), 10 * sizeof(float), cudaMemcpyDeviceToHost);
    print_values(v2, "GPU: Second generation (advanced state)");
    
    // Restore state to state1 and regenerate
    RNG::set_state(state1);
    Tensor t3 = Tensor::rand<float>(Shape{{10}}, TensorOptions().with_device(Device::CUDA));
    std::vector<float> v3(10);
    cudaMemcpy(v3.data(), t3.data(), 10 * sizeof(float), cudaMemcpyDeviceToHost);
    print_values(v3, "GPU: Third generation (restored state)");
    
    // Verify t1 == t3
    assert(compare_vectors(v1, v3) && "GPU: Restored state should reproduce same values");
    assert(!compare_vectors(v1, v2) && "GPU: Advanced state should produce different values");
    
    std::cout << "✓ TEST 6 PASSED: GPU state restore works correctly\n";
}

// ==============================================================================
// TEST 7: GPU RNG Multiple State Snapshots
// ==============================================================================
void test_gpu_rng_multiple_snapshots() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST 7: GPU RNG Multiple State Snapshots\n";
    std::cout << std::string(80, '=') << "\n";

    RNG::set_seed(123);
    
    std::vector<RNGState> snapshots;
    std::vector<std::vector<float>> results;
    
    // Take 5 snapshots and generate tensors
    for (int i = 0; i < 5; i++) {
        snapshots.push_back(RNG::get_state());
        
        Tensor t = Tensor::rand<float>(Shape{{5}}, TensorOptions().with_device(Device::CUDA));
        std::vector<float> v(5);
        cudaMemcpy(v.data(), t.data(), 5 * sizeof(float), cudaMemcpyDeviceToHost);
        results.push_back(v);
        
        std::cout << "GPU Snapshot " << i << " generated: ";
        print_values(v, "");
    }
    
    // Now restore each snapshot and verify we get same results
    std::cout << "\nVerifying GPU snapshot restoration:\n";
    for (int i = 0; i < 5; i++) {
        RNG::set_state(snapshots[i]);
        Tensor t = Tensor::rand<float>(Shape{{5}}, TensorOptions().with_device(Device::CUDA));
        std::vector<float> v(5);
        cudaMemcpy(v.data(), t.data(), 5 * sizeof(float), cudaMemcpyDeviceToHost);
        
        std::cout << "GPU Snapshot " << i << " restored:  ";
        print_values(v, "");
        
        assert(compare_vectors(results[i], v) && "GPU: Snapshot restoration failed");
    }
    
    std::cout << "✓ TEST 7 PASSED: GPU Multiple snapshots work correctly\n";
}

// ==============================================================================
// TEST 8: GPU RNG - randn (Normal Distribution)
// ==============================================================================
void test_gpu_rng_normal_distribution() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST 8: GPU RNG - Normal Distribution State\n";
    std::cout << std::string(80, '=') << "\n";

    RNG::set_seed(789);
    
    // Capture state
    RNGState state = RNG::get_state();
    
    // Generate normal distribution
    Tensor t1 = Tensor::randn<float>(Shape{{15}}, TensorOptions().with_device(Device::CUDA), 0, 2.0f);
    std::vector<float> v1(15);
    cudaMemcpy(v1.data(), t1.data(), 15 * sizeof(float), cudaMemcpyDeviceToHost);
    print_values(v1, "GPU Normal dist (σ=2.0)");
    
    // Restore and regenerate
    RNG::set_state(state);
    Tensor t2 = Tensor::randn<float>(Shape{{15}}, TensorOptions().with_device(Device::CUDA), 0, 2.0f);
    std::vector<float> v2(15);
    cudaMemcpy(v2.data(), t2.data(), 15 * sizeof(float), cudaMemcpyDeviceToHost);
    print_values(v2, "GPU Normal dist (restored)");
    
    assert(compare_vectors(v1, v2) && "GPU: Normal distribution state restoration failed");
    
    std::cout << "✓ TEST 8 PASSED: GPU Normal distribution state works correctly\n";
}

// ==============================================================================
// TEST 9: GPU RNG - Seeding Effects
// ==============================================================================
void test_gpu_rng_seeding() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST 9: GPU RNG - Seeding Effects\n";
    std::cout << std::string(80, '=') << "\n";

    // Same seed should produce same results
    RNG::set_seed(999);
    Tensor t1 = Tensor::rand<float>(Shape{{8}}, TensorOptions().with_device(Device::CUDA));
    std::vector<float> v1(8);
    cudaMemcpy(v1.data(), t1.data(), 8 * sizeof(float), cudaMemcpyDeviceToHost);
    print_values(v1, "GPU Seed 999 - first run");
    
    RNG::set_seed(999);
    Tensor t2 = Tensor::rand<float>(Shape{{8}}, TensorOptions().with_device(Device::CUDA));
    std::vector<float> v2(8);
    cudaMemcpy(v2.data(), t2.data(), 8 * sizeof(float), cudaMemcpyDeviceToHost);
    print_values(v2, "GPU Seed 999 - second run");
    
    assert(compare_vectors(v1, v2) && "GPU: Same seed should produce same results");
    
    // Different seed should produce different results
    RNG::set_seed(1000);
    Tensor t3 = Tensor::rand<float>(Shape{{8}}, TensorOptions().with_device(Device::CUDA));
    std::vector<float> v3(8);
    cudaMemcpy(v3.data(), t3.data(), 8 * sizeof(float), cudaMemcpyDeviceToHost);
    print_values(v3, "GPU Seed 1000");
    
    assert(!compare_vectors(v1, v3) && "GPU: Different seed should produce different results");
    
    std::cout << "✓ TEST 9 PASSED: GPU Seeding works correctly\n";
}

// ==============================================================================
// TEST 10: GPU RNG - Mixed rand and randn
// ==============================================================================
void test_gpu_rng_mixed_operations() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST 10: GPU RNG - Mixed rand and randn Operations\n";
    std::cout << std::string(80, '=') << "\n";

    RNG::set_seed(555);
    
    // Capture state
    RNGState state = RNG::get_state();
    
    // Generate sequence: rand -> randn -> rand
    Tensor t1_uniform = Tensor::rand<float>(Shape{{5}}, TensorOptions().with_device(Device::CUDA));
    Tensor t1_normal = Tensor::randn<float>(Shape{{5}}, TensorOptions().with_device(Device::CUDA));
    Tensor t1_uniform2 = Tensor::rand<float>(Shape{{5}}, TensorOptions().with_device(Device::CUDA));
    
    std::vector<float> v1_u1(5), v1_n(5), v1_u2(5);
    cudaMemcpy(v1_u1.data(), t1_uniform.data(), 5 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(v1_n.data(), t1_normal.data(), 5 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(v1_u2.data(), t1_uniform2.data(), 5 * sizeof(float), cudaMemcpyDeviceToHost);
    
    print_values(v1_u1, "GPU Sequence 1: Uniform");
    print_values(v1_n, "GPU Sequence 1: Normal");
    print_values(v1_u2, "GPU Sequence 1: Uniform again");
    
    // Restore state and repeat
    RNG::set_state(state);
    
    Tensor t2_uniform = Tensor::rand<float>(Shape{{5}}, TensorOptions().with_device(Device::CUDA));
    Tensor t2_normal = Tensor::randn<float>(Shape{{5}}, TensorOptions().with_device(Device::CUDA));
    Tensor t2_uniform2 = Tensor::rand<float>(Shape{{5}}, TensorOptions().with_device(Device::CUDA));
    
    std::vector<float> v2_u1(5), v2_n(5), v2_u2(5);
    cudaMemcpy(v2_u1.data(), t2_uniform.data(), 5 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(v2_n.data(), t2_normal.data(), 5 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(v2_u2.data(), t2_uniform2.data(), 5 * sizeof(float), cudaMemcpyDeviceToHost);
    
    print_values(v2_u1, "GPU Sequence 2: Uniform");
    print_values(v2_n, "GPU Sequence 2: Normal");
    print_values(v2_u2, "GPU Sequence 2: Uniform again");
    
    assert(compare_vectors(v1_u1, v2_u1) && "GPU: First uniform failed");
    assert(compare_vectors(v1_n, v2_n) && "GPU: Normal failed");
    assert(compare_vectors(v1_u2, v2_u2) && "GPU: Second uniform failed");
    
    std::cout << "✓ TEST 10 PASSED: GPU Mixed operations state works correctly\n";
}

// ==============================================================================
// TEST 11: CPU vs GPU - Same Seed Comparison
// ==============================================================================
void test_cpu_gpu_seed_comparison() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST 11: CPU vs GPU - Same Seed Comparison\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "(Note: CPU and GPU use different RNG algorithms, so values will differ)\n\n";

    const unsigned long seed = 12345;
    
    // CPU generation
    RNG::set_seed(seed);
    Tensor t_cpu = Tensor::rand<float>(Shape{{10}}, TensorOptions().with_device(Device::CPU));
    std::vector<float> v_cpu(10);
    std::copy(t_cpu.data<float>(), t_cpu.data<float>() + 10, v_cpu.begin());
    print_values(v_cpu, "CPU (seed 12345)");
    
    // GPU generation
    RNG::set_seed(seed);
    Tensor t_gpu = Tensor::rand<float>(Shape{{10}}, TensorOptions().with_device(Device::CUDA));
    std::vector<float> v_gpu(10);
    cudaMemcpy(v_gpu.data(), t_gpu.data(), 10 * sizeof(float), cudaMemcpyDeviceToHost);
    print_values(v_gpu, "GPU (seed 12345)");
    
    // They should be different (different algorithms)
    bool are_different = !compare_vectors(v_cpu, v_gpu, 0.1f);
    std::cout << "\nCPU and GPU values are " << (are_different ? "DIFFERENT" : "SAME") 
              << " (as expected, different RNG algorithms)\n";
    
    std::cout << "✓ TEST 11 PASSED: CPU/GPU comparison documented\n";
}

// ==============================================================================
// TEST 12: GPU RNG with Double Precision
// ==============================================================================
void test_gpu_rng_double_precision() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST 12: GPU RNG with Double Precision\n";
    std::cout << std::string(80, '=') << "\n";

    RNG::set_seed(777);
    
    // Capture state
    RNGState state = RNG::get_state();
    
    // Generate double precision tensor
    Tensor t1 = Tensor::rand<double>(Shape{{8}}, TensorOptions().with_device(Device::CUDA).with_dtype(Dtype::Float32));
    std::vector<double> v1(8);
    cudaMemcpy(v1.data(), t1.data(), 8 * sizeof(double), cudaMemcpyDeviceToHost);
    
    std::cout << "GPU Double precision: ";
    for (const auto& val : v1) {
        std::cout << std::fixed << std::setprecision(12) << val << " ";
    }
    std::cout << "\n";
    
    // Restore and regenerate
    RNG::set_state(state);
    Tensor t2 = Tensor::rand<double>(Shape{{8}}, TensorOptions().with_device(Device::CUDA).with_dtype(Dtype::Float32));
    std::vector<double> v2(8);
    cudaMemcpy(v2.data(), t2.data(), 8 * sizeof(double), cudaMemcpyDeviceToHost);
    
    std::cout << "GPU Double (restored):  ";
    for (const auto& val : v2) {
        std::cout << std::fixed << std::setprecision(12) << val << " ";
    }
    std::cout << "\n";
    
    // Compare
    bool match = true;
    for (size_t i = 0; i < v1.size(); ++i) {
        if (std::abs(v1[i] - v2[i]) > 1e-12) {
            match = false;
            break;
        }
    }
    
    assert(match && "GPU: Double precision state restoration failed");
    
    std::cout << "✓ TEST 12 PASSED: GPU Double precision state works correctly\n";
}
#endif

// ==============================================================================
// TEST 13: RNGStateGuard RAII Test (CPU)
// ==============================================================================
void test_rng_state_guard_cpu() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST 13: RNGStateGuard RAII Test (CPU)\n";
    std::cout << std::string(80, '=') << "\n";

    RNG::set_seed(333);
    
    // Generate baseline
    Tensor t1 = Tensor::rand<float>(Shape{{6}}, TensorOptions().with_device(Device::CPU));
    std::vector<float> v1(6);
    std::copy(t1.data<float>(), t1.data<float>() + 6, v1.begin());
    print_values(v1, "Before guard scope");
    
    // Use RNGStateGuard
    {
        RNGStateGuard guard;
        
        // Generate inside guard scope
        Tensor t2 = Tensor::rand<float>(Shape{{6}}, TensorOptions().with_device(Device::CPU));
        std::vector<float> v2(6);
        std::copy(t2.data<float>(), t2.data<float>() + 6, v2.begin());
        print_values(v2, "Inside guard scope");
        
        // This should advance the state
        Tensor t3 = Tensor::rand<float>(Shape{{6}}, TensorOptions().with_device(Device::CPU));
        std::vector<float> v3(6);
        std::copy(t3.data<float>(), t3.data<float>() + 6, v3.begin());
        print_values(v3, "Still inside guard");
    }
    // Guard destructor should restore state here
    
    // This should match t2 (first generation inside guard)
    Tensor t4 = Tensor::rand<float>(Shape{{6}}, TensorOptions().with_device(Device::CPU));
    std::vector<float> v4(6);
    std::copy(t4.data<float>(), t4.data<float>() + 6, v4.begin());
    print_values(v4, "After guard scope (should match 'Inside guard')");
    
    // Read v2 again for comparison (we need to regenerate it)
    RNG::set_seed(333);
    Tensor t_base = Tensor::rand<float>(Shape{{6}}, TensorOptions().with_device(Device::CPU));
    Tensor t_expected = Tensor::rand<float>(Shape{{6}}, TensorOptions().with_device(Device::CPU));
    std::vector<float> v_expected(6);
    std::copy(t_expected.data<float>(), t_expected.data<float>() + 6, v_expected.begin());
    
    assert(compare_vectors(v4, v_expected) && "RNGStateGuard failed to restore state");
    
    std::cout << "✓ TEST 13 PASSED: RNGStateGuard RAII works correctly\n";
}

// ==============================================================================
// MAIN
// ==============================================================================
int main() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         COMPREHENSIVE RNG STATE TESTS - CPU AND GPU                      ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════════════╝\n";

    try {
        // CPU Tests
        test_cpu_rng_basic_state();              // TEST 1
        test_cpu_rng_multiple_snapshots();        // TEST 2
        test_cpu_rng_normal_distribution();       // TEST 3
        test_cpu_rng_seeding();                   // TEST 4
        test_cpu_rng_mixed_operations();          // TEST 5
        
#ifdef WITH_CUDA
        // GPU Tests
        test_gpu_rng_basic_state();               // TEST 6
        test_gpu_rng_multiple_snapshots();        // TEST 7
        test_gpu_rng_normal_distribution();       // TEST 8
        test_gpu_rng_seeding();                   // TEST 9
        test_gpu_rng_mixed_operations();          // TEST 10
        test_cpu_gpu_seed_comparison();           // TEST 11
        test_gpu_rng_double_precision();          // TEST 12
#endif
        
        // RAII Test
        test_rng_state_guard_cpu();               // TEST 13
        
        std::cout << "\n";
        std::cout << "╔═══════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                     ALL TESTS PASSED! ✓✓✓                                ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════════════════════╝\n";
        std::cout << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ TEST FAILED: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}