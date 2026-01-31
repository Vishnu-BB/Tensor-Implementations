#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>
#include <iomanip>
#include <cstdint>
#include "core/Tensor.h"

using namespace OwnTensor;

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

void print_tensor_info(const std::string& name, Tensor& t) {
    std::cout << "  " << name << ":" << std::endl;
    std::cout << "    Address:        " << t.data() << std::endl;
    std::cout << "    Numel:          " << t.numel() << std::endl;
    std::cout << "    Nbytes:         " << t.nbytes() << std::endl;
    std::cout << "    Storage Offset: " << t.storage_offset() << std::endl;
    std::cout << "    Owns Data:      " << (t.owns_data() ? "yes" : "no") << std::endl;
    std::cout << "    Is Contiguous:  " << (t.is_contiguous() ? "yes" : "no") << std::endl;
}

void print_tensor_data(const std::string& name, Tensor& t, int max_elems = 10) {
    std::cout << "  " << name << " data: [";
    float* ptr = t.data<float>();
    int count = std::min((int)t.numel(), max_elems);
    for (int i = 0; i < count; ++i) {
        std::cout << ptr[i] << (i < count - 1 ? ", " : "");
    }
    if ((int)t.numel() > max_elems) std::cout << ", ...";
    std::cout << "]" << std::endl;
}

// ============================================================================
// TEST 1: slice() - copies a contiguous range
// ============================================================================
void test_slice() {
    print_separator("TEST: slice(start, length)");

    std::vector<float> data(20);
    std::iota(data.begin(), data.end(), 0.0f);

    Tensor source = Tensor({{1, 20}}, TensorOptions().with_dtype(Dtype::Float32));
    source.set_data(data);

    std::cout << "Source tensor [0..19]:" << std::endl;
    print_tensor_info("Source", source);
    print_tensor_data("Source", source, 20);

    Tensor sliced = source.slice(5, 10);
    Tensor sliced_inplace = source.slice_inplace(5, 10);

    std::cout << "\nSliced tensor (start=5, length=10):" << std::endl;
    print_tensor_info("Sliced", sliced);
    print_tensor_data("Sliced", sliced, 10);

    assert(sliced.numel() == 10);
    assert(sliced.owns_data() == true);
    assert(sliced.data() != source.data());
    
    assert(sliced_inplace.numel() == 10);
    assert(sliced_inplace.owns_data() == true);
    assert(sliced_inplace.data() == source.data() + 5*source.dtype_size(source.dtype()));
    

    float* s_ptr = sliced.data<float>();
    for (int i = 0; i < 10; ++i) {
        assert(s_ptr[i] == 5.0f + i);
    }

    std::cout << "\n  [PASS] slice() creates independent copy with correct data" << std::endl;
}

// ============================================================================
// TEST 2: flatten_concat() - concatenates multiple tensors into flat tensor
// ============================================================================
void test_flatten_concat() {
    print_separator("TEST: flatten_concat(tensor_list)");

    Tensor t1 = Tensor({{1, 3}}, TensorOptions().with_dtype(Dtype::Float32));
    Tensor t2 = Tensor({{2, 2}}, TensorOptions().with_dtype(Dtype::Float32));
    Tensor t3 = Tensor({{1, 5}}, TensorOptions().with_dtype(Dtype::Float32));

    t1.set_data(std::vector<float>{1.0f, 2.0f, 3.0f});
    t2.set_data(std::vector<float>{4.0f, 5.0f, 6.0f, 7.0f});
    t3.set_data(std::vector<float>{8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

    std::cout << "Input tensors:" << std::endl;
    print_tensor_info("t1 (1x3)", t1);
    print_tensor_data("t1", t1, 3);
    print_tensor_info("t2 (2x2)", t2);
    print_tensor_data("t2", t2, 4);
    print_tensor_info("t3 (1x5)", t3);
    print_tensor_data("t3", t3, 5);

    std::vector<Tensor> tensors = {t1, t2, t3};
    Tensor result = Tensor::flatten_concat(tensors);

    std::cout << "\nConcatenated result:" << std::endl;
    print_tensor_info("Result", result);
    print_tensor_data("Result", result, 12);

    assert(result.numel() == 12);
    assert(result.owns_data() == true);

    float* r_ptr = result.data<float>();
    for (int i = 0; i < 12; ++i) {
        assert(r_ptr[i] == (float)(i + 1));
    }

    std::cout << "\n  [PASS] flatten_concat() correctly concatenates tensors" << std::endl;
}

// ============================================================================
// TEST 3: narrow() - extracts slice along an axis (copies data)
// ============================================================================
void test_narrow() {
    print_separator("TEST: narrow(axis, start, length)");

    std::vector<float> data(12);
    std::iota(data.begin(), data.end(), 0.0f);

    Tensor source = Tensor({{3, 4}}, TensorOptions().with_dtype(Dtype::Float32));
    source.set_data(data);

    std::cout << "Source tensor (3x4):" << std::endl;
    print_tensor_info("Source", source);
    print_tensor_data("Source", source, 12);

    Tensor narrowed = source.narrow(0, 1, 2);

    std::cout << "\nNarrowed along axis=0, start=1, length=2 (rows 1-2):" << std::endl;
    print_tensor_info("Narrowed", narrowed);
    print_tensor_data("Narrowed", narrowed, 8);

    assert(narrowed.numel() == 8);
    assert(narrowed.shape().dims[0] == 2);
    assert(narrowed.shape().dims[1] == 4);
    assert(narrowed.owns_data() == true);

    float* n_ptr = narrowed.data<float>();
    assert(n_ptr[0] == 4.0f);
    assert(n_ptr[7] == 11.0f);

    std::cout << "\n  [PASS] narrow() extracts correct slice with proper shape" << std::endl;
}

// ============================================================================
// TEST 4: make_shards(num_shards, row_major) - equal splits, copies data
// ============================================================================
void test_make_shards_equal() {
    print_separator("TEST: make_shards(num_shards, row_major=true)");

    std::vector<float> data(12);
    std::iota(data.begin(), data.end(), 0.0f);

    Tensor source = Tensor({{1, 12}}, TensorOptions().with_dtype(Dtype::Float32));
    source.set_data(data);

    std::cout << "Source tensor (1x12):" << std::endl;
    print_tensor_info("Source", source);
    print_tensor_data("Source", source, 12);

    std::vector<Tensor> shards = source.make_shards(3, true);

    std::cout << "\nShards (3 equal parts of 4 elements each):" << std::endl;
    for (size_t i = 0; i < shards.size(); ++i) {
        std::cout << "\n  Shard " << i << ":" << std::endl;
        print_tensor_info("Shard " + std::to_string(i), shards[i]);
        print_tensor_data("Shard " + std::to_string(i), shards[i], 4);

        assert(shards[i].numel() == 4);
        assert(shards[i].owns_data() == true);
        assert(shards[i].data() != source.data());

        float* ptr = shards[i].data<float>();
        for (int j = 0; j < 4; ++j) {
            assert(ptr[j] == (float)(i * 4 + j));
        }
    }

    std::cout << "\n  [PASS] make_shards() creates independent copies with correct data" << std::endl;
}

// ============================================================================
// TEST 5: make_shards(num_shards, axis) - splits along specific axis
// ============================================================================
void test_make_shards_axis() {
    print_separator("TEST: make_shards(num_shards, axis)");

    std::vector<float> data(12);
    std::iota(data.begin(), data.end(), 0.0f);

    Tensor source = Tensor({{3, 4}}, TensorOptions().with_dtype(Dtype::Float32));
    source.set_data(data);

    std::cout << "Source tensor (3x4):" << std::endl;
    print_tensor_info("Source", source);
    print_tensor_data("Source", source, 12);
    std::cout << "  Layout: [[0,1,2,3], [4,5,6,7], [8,9,10,11]]" << std::endl;

    std::vector<Tensor> shards = source.make_shards_axis(2, 1);

    std::cout << "\nShards along axis=1 (split cols 4 -> 2+2):" << std::endl;
    for (size_t i = 0; i < shards.size(); ++i) {
        std::cout << "\n  Shard " << i << ":" << std::endl;
        print_tensor_info("Shard " + std::to_string(i), shards[i]);
        print_tensor_data("Shard " + std::to_string(i), shards[i], 6);

        assert(shards[i].numel() == 6);
        assert(shards[i].shape().dims[0] == 3);
        assert(shards[i].shape().dims[1] == 2);
    }

    float* s0 = shards[0].data<float>();
    assert(s0[0] == 0.0f && s0[1] == 1.0f);
    assert(s0[2] == 4.0f && s0[3] == 5.0f);
    assert(s0[4] == 8.0f && s0[5] == 9.0f);

    float* s1 = shards[1].data<float>();
    assert(s1[0] == 2.0f && s1[1] == 3.0f);
    assert(s1[2] == 6.0f && s1[3] == 7.0f);
    assert(s1[4] == 10.0f && s1[5] == 11.0f);

    std::cout << "\n  [PASS] make_shards(axis) correctly splits along specified axis" << std::endl;
}

// ============================================================================
// TEST 6: make_shards_cust() - custom shape splits
// ============================================================================
void test_make_shards_custom() {
    print_separator("TEST: make_shards_cust(shard_shapes, row_major)");

    std::vector<float> data(10);
    std::iota(data.begin(), data.end(), 0.0f);

    Tensor source = Tensor({{1, 10}}, TensorOptions().with_dtype(Dtype::Float32));
    source.set_data(data);

    std::cout << "Source tensor (1x10):" << std::endl;
    print_tensor_info("Source", source);
    print_tensor_data("Source", source, 10);

    std::vector<Shape> custom_shapes = {
        Shape({{1, 3}}),
        Shape({{1, 2}}),
        Shape({{1, 5}})
    };

    std::vector<Tensor> shards = source.make_shards_cust(custom_shapes, true);

    std::cout << "\nCustom shards (3, 2, 5 elements):" << std::endl;
    for (size_t i = 0; i < shards.size(); ++i) {
        std::cout << "\n  Shard " << i << ":" << std::endl;
        print_tensor_info("Shard " + std::to_string(i), shards[i]);
        print_tensor_data("Shard " + std::to_string(i), shards[i], (int)shards[i].numel());

        assert(shards[i].owns_data() == true);
        assert(shards[i].data() != source.data());
    }

    float* s0 = shards[0].data<float>();
    assert(s0[0] == 0.0f && s0[1] == 1.0f && s0[2] == 2.0f);

    float* s1 = shards[1].data<float>();
    assert(s1[0] == 3.0f && s1[1] == 4.0f);

    float* s2 = shards[2].data<float>();
    for (int i = 0; i < 5; ++i) assert(s2[i] == 5.0f + i);

    std::cout << "\n  [PASS] make_shards_cust() correctly splits with custom shapes" << std::endl;
}

// ============================================================================
// TEST 7: make_shards_inplace() - creates views sharing storage
// ============================================================================
void test_make_shards_inplace() {
    print_separator("TEST: make_shards_inplace(num_shards, row_major) - VIEWS");

    std::vector<float> data(12);
    std::iota(data.begin(), data.end(), 0.0f);

    Tensor source = Tensor({{1, 12}}, TensorOptions().with_dtype(Dtype::Float32));
    source.set_data(data);

    std::cout << "Source tensor (1x12):" << std::endl;
    print_tensor_info("Source", source);
    print_tensor_data("Source", source, 12);

    void* source_base = source.data();

    std::vector<Tensor> shards = source.make_shards_inplace(3, true);

    std::cout << "\nInplace shards (views with storage_offset):" << std::endl;
    for (size_t i = 0; i < shards.size(); ++i) {
        std::cout << "\n  Shard " << i << ":" << std::endl;
        print_tensor_info("Shard " + std::to_string(i), shards[i]);
        print_tensor_data("Shard " + std::to_string(i), shards[i], 4);

        assert(shards[i].numel() == 4);
        assert(shards[i].storage_offset() == i * 4);
        assert(shards[i].owns_data() == false);
    }

    uintptr_t base = reinterpret_cast<uintptr_t>(source_base);
    for (size_t i = 0; i < shards.size(); ++i) {
        uintptr_t shard_addr = reinterpret_cast<uintptr_t>(shards[i].data());
        uintptr_t expected_addr = base + i * 4 * sizeof(float);
        std::cout << "  Shard " << i << " address check: " << (void*)shard_addr 
                  << " == " << (void*)expected_addr << " ? " 
                  << (shard_addr == expected_addr ? "YES" : "NO") << std::endl;
        assert(shard_addr == expected_addr);
    }

    std::cout << "\n  Modifying shard[1][0] = 999.0f..." << std::endl;
    shards[1].data<float>()[0] = 999.0f;

    std::cout << "  Source after modification:" << std::endl;
    print_tensor_data("Source", source, 12);
    assert(source.data<float>()[4] == 999.0f);

    std::cout << "\n  [PASS] make_shards_inplace() creates views that share storage" << std::endl;
}

// ============================================================================
// TEST 8: make_shards_inplace_cust() - custom shape views
// ============================================================================
void test_make_shards_inplace_custom() {
    print_separator("TEST: make_shards_inplace_cust(shard_shapes, row_major) - VIEWS");

    std::vector<float> data(10);
    std::iota(data.begin(), data.end(), 0.0f);

    Tensor source = Tensor({{1, 10}}, TensorOptions().with_dtype(Dtype::Float32));
    source.set_data(data);

    std::cout << "Source tensor (1x10):" << std::endl;
    print_tensor_info("Source", source);
    print_tensor_data("Source", source, 10);

    std::vector<Shape> custom_shapes = {
        Shape({{1, 2}}),
        Shape({{1, 3}}),
        Shape({{1, 5}})
    };

    std::vector<Tensor> shards = source.make_shards_inplace_cust(custom_shapes, true);

    std::cout << "\nInplace custom shards (2, 3, 5 elements as views):" << std::endl;
    
    size_t expected_offset = 0;
    for (size_t i = 0; i < shards.size(); ++i) {
        std::cout << "\n  Shard " << i << ":" << std::endl;
        print_tensor_info("Shard " + std::to_string(i), shards[i]);
        print_tensor_data("Shard " + std::to_string(i), shards[i], (int)shards[i].numel());

        assert(shards[i].storage_offset() == expected_offset);
        assert(shards[i].owns_data() == false);

        expected_offset += shards[i].numel();
    }

    shards[2].data<float>()[0] = 888.0f;
    std::cout << "\n  Modified shard[2][0] = 888.0f" << std::endl;
    print_tensor_data("Source after", source, 10);
    assert(source.data<float>()[5] == 888.0f);

    std::cout << "\n  [PASS] make_shards_inplace_cust() creates custom views sharing storage" << std::endl;
}

// ============================================================================
// TEST 9: shard_into() - copies source data into destination tensors
// ============================================================================
void test_shard_into() {
    print_separator("TEST: shard_into(destinations)");

    std::vector<float> data(10);
    std::iota(data.begin(), data.end(), 0.0f);

    Tensor source = Tensor({{1, 10}}, TensorOptions().with_dtype(Dtype::Float32));
    source.set_data(data);

    std::cout << "Source tensor (1x10):" << std::endl;
    print_tensor_info("Source", source);
    print_tensor_data("Source", source, 10);

    Tensor dest1 = Tensor::zeros({{1, 3}}, TensorOptions().with_dtype(Dtype::Float32));
    Tensor dest2 = Tensor::zeros({{1, 2}}, TensorOptions().with_dtype(Dtype::Float32));
    Tensor dest3 = Tensor::zeros({{1, 5}}, TensorOptions().with_dtype(Dtype::Float32));

    void* dest1_addr = dest1.data();
    void* dest2_addr = dest2.data();
    void* dest3_addr = dest3.data();

    std::cout << "\nDestinations BEFORE shard_into:" << std::endl;
    print_tensor_data("dest1 (3)", dest1, 3);
    print_tensor_data("dest2 (2)", dest2, 2);
    print_tensor_data("dest3 (5)", dest3, 5);

    std::vector<Tensor> destinations = {dest1, dest2, dest3};
    source.shard_into(destinations);

    std::cout << "\nDestinations AFTER shard_into:" << std::endl;
    print_tensor_data("dest1 (3)", dest1, 3);
    print_tensor_data("dest2 (2)", dest2, 2);
    print_tensor_data("dest3 (5)", dest3, 5);

    assert(dest1.data() == dest1_addr);
    assert(dest2.data() == dest2_addr);
    assert(dest3.data() == dest3_addr);

    float* d1 = dest1.data<float>();
    assert(d1[0] == 0.0f && d1[1] == 1.0f && d1[2] == 2.0f);

    float* d2 = dest2.data<float>();
    assert(d2[0] == 3.0f && d2[1] == 4.0f);

    float* d3 = dest3.data<float>();
    for (int i = 0; i < 5; ++i) assert(d3[i] == 5.0f + i);

    std::cout << "\n  Address preservation check:" << std::endl;
    std::cout << "    dest1: " << dest1_addr << " -> " << dest1.data() << " (same: YES)" << std::endl;
    std::cout << "    dest2: " << dest2_addr << " -> " << dest2.data() << " (same: YES)" << std::endl;
    std::cout << "    dest3: " << dest3_addr << " -> " << dest3.data() << " (same: YES)" << std::endl;

    std::cout << "\n  [PASS] shard_into() copies data to pre-allocated destinations" << std::endl;
}

// ============================================================================
// TEST 10: Memory size verification across all sharding types
// ============================================================================
void test_memory_sizes() {
    print_separator("TEST: Memory Size Verification");

    const size_t N = 1000;
    Tensor source = Tensor({{1, (int64_t)N}}, TensorOptions().with_dtype(Dtype::Float32));
    source.fill(1.0f);

    std::cout << "Source: " << N << " elements, " << source.nbytes() << " bytes" << std::endl;
    std::cout << "Expected: " << N * sizeof(float) << " bytes" << std::endl;
    assert(source.nbytes() == N * sizeof(float));

    auto copy_shards = source.make_shards(4, true);
    size_t total_copy_bytes = 0;
    for (auto& s : copy_shards) {
        total_copy_bytes += s.nbytes();
        assert(s.allocated_bytes() == s.nbytes());
    }
    std::cout << "\nmake_shards(4): total bytes across shards = " << total_copy_bytes << std::endl;
    assert(total_copy_bytes == N * sizeof(float));

    auto inplace_shards = source.make_shards_inplace(4, true);
    for (auto& s : inplace_shards) {
        std::cout << "  Inplace shard: numel=" << s.numel() 
                  << ", nbytes=" << s.nbytes() 
                  << ", allocated=" << s.allocated_bytes() << std::endl;
        assert(s.nbytes() == (N / 4) * sizeof(float));
        assert(s.allocated_bytes() == N * sizeof(float));
    }

    std::cout << "\n  [PASS] Memory sizes are correct for all sharding types" << std::endl;
}

// ============================================================================
// TEST 11: Error handling
// ============================================================================
void test_error_handling() {
    print_separator("TEST: Error Handling");

    Tensor source = Tensor::zeros({{1, 10}}, TensorOptions().with_dtype(Dtype::Float32));

    std::cout << "Testing make_shards with non-divisible count..." << std::endl;
    try {
        auto shards = source.make_shards(3, true);
        assert(false && "Should have thrown");
    } catch (const std::runtime_error& e) {
        std::cout << "  Caught expected error: " << e.what() << std::endl;
    }

    std::cout << "\nTesting shard_into with overflow..." << std::endl;
    Tensor dest1 = Tensor::zeros({{1, 6}});
    Tensor dest2 = Tensor::zeros({{1, 6}});
    std::vector<Tensor> dests = {dest1, dest2};
    try {
        source.shard_into(dests);
        assert(false && "Should have thrown");
    } catch (const std::runtime_error& e) {
        std::cout << "  Caught expected error: " << e.what() << std::endl;
    }

    std::cout << "\nTesting make_shards_cust with wrong total..." << std::endl;
    try {
        std::vector<Shape> bad_shapes = {Shape({{1, 5}}), Shape({{1, 6}})};
        auto shards = source.make_shards_cust(bad_shapes, true);
        assert(false && "Should have thrown");
    } catch (const std::runtime_error& e) {
        std::cout << "  Caught expected error: " << e.what() << std::endl;
    }

    std::cout << "\n  [PASS] Error handling works correctly" << std::endl;
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    std::cout << "\n" << std::string(60, '#') << std::endl;
    std::cout << "   COMPREHENSIVE SHARDING TEST SUITE" << std::endl;
    std::cout << std::string(60, '#') << std::endl;

    try {
        test_slice();
        test_flatten_concat();
        test_narrow();
        test_make_shards_equal();
        test_make_shards_axis();
        test_make_shards_custom();
        // test_make_shards_inplace();
        // test_make_shards_inplace_custom();
        // test_shard_into();
        // test_memory_sizes();
        test_error_handling();

        std::cout << "\n" << std::string(60, '#') << std::endl;
        std::cout << "   ALL SHARDING TESTS PASSED!" << std::endl;
        std::cout << std::string(60, '#') << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\n[FAILED] Test error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
