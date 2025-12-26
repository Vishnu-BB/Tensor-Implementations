#include "TensorLib.h"
#include <iostream>
#include <cassert>
#include <vector>

using namespace OwnTensor;
using namespace std;

void test_set_grad_float() {
    cout << "Testing set_grad<float>..." << endl;
    Tensor t(Shape{{4}}, TensorOptions().with_dtype(Dtype::Float32).with_req_grad(true));
    vector<float> grad_vals = {1.0f, 2.0f, 3.0f, 4.0f};
    t.set_grad(grad_vals);
    
    // grad() returns void*, need to cast to float*
    float* g_ptr = reinterpret_cast<float*>(t.grad());
    for (size_t i = 0; i < 4; ++i) {
        assert(g_ptr[i] == grad_vals[i]);
    }
    cout << "✓ set_grad<float> passed" << endl;
}

void test_fill_grad_float() {
    cout << "Testing fill_grad<float>..." << endl;
    Tensor t(Shape{{4}}, TensorOptions().with_dtype(Dtype::Float32).with_req_grad(true));
    t.fill_grad(5.0f);
    
    float* g_ptr = reinterpret_cast<float*>(t.grad());
    for (size_t i = 0; i < 4; ++i) {
        assert(g_ptr[i] == 5.0f);
    }
    cout << "✓ fill_grad<float> passed" << endl;
}

void test_set_grad_float16() {
    cout << "Testing set_grad<float16_t>..." << endl;
    Tensor t(Shape{{2}}, TensorOptions().with_dtype(Dtype::Float16).with_req_grad(true));
    vector<float16_t> grad_vals = {float16_t(1.5f), float16_t(2.5f)};
    t.set_grad(grad_vals);
    
    // Gradient is stored as uint16_t raw bits on CPU
    uint16_t* g_ptr = reinterpret_cast<uint16_t*>(t.grad());
    assert(g_ptr[0] == grad_vals[0].raw_bits);
    assert(g_ptr[1] == grad_vals[1].raw_bits);
    cout << "✓ set_grad<float16_t> passed" << endl;
}

void test_set_grad_bfloat16() {
    cout << "Testing set_grad<bfloat16_t>..." << endl;
    Tensor t(Shape{{2}}, TensorOptions().with_dtype(Dtype::Bfloat16).with_req_grad(true));
    vector<bfloat16_t> grad_vals = {bfloat16_t(1.1f), bfloat16_t(2.2f)};
    t.set_grad(grad_vals);
    
    uint16_t* g_ptr = reinterpret_cast<uint16_t*>(t.grad());
    assert(g_ptr[0] == grad_vals[0].raw_bits);
    assert(g_ptr[1] == grad_vals[1].raw_bits);
    cout << "✓ set_grad<bfloat16_t> passed" << endl;
}

int main() {
    try {
        test_set_grad_float();
        test_fill_grad_float();
        test_set_grad_float16();
        test_set_grad_bfloat16();
        cout << "\nALL GRAD MANIP TESTS PASSED! ✓" << endl;
    } catch (const exception& e) {
        cerr << "Test failed with exception: " << e.what() << endl;
        return 1;
    }
    return 0;
}
