#include <iostream>
#include <cassert>
#include "core/Tensor.h"
#include "dtype/DtypeTraits.h"

using namespace OwnTensor;

void test_promotion(Dtype a, Dtype b, Dtype expected, const std::string& name) {
    Dtype result = promote_dtypes_bool(a, b);
    if (result == expected) {
        std::cout << "[PASS] " << name << ": " 
                  << get_dtype_name(a) << " + " << get_dtype_name(b) 
                  << " -> " << get_dtype_name(result) << std::endl;
    } else {
        std::cout << "[FAIL] " << name << ": " 
                  << get_dtype_name(a) << " + " << get_dtype_name(b) 
                  << " -> " << get_dtype_name(result) 
                  << " (Expected: " << get_dtype_name(expected) << ")" << std::endl;
        exit(1);
    }
}

int main() {
    std::cout << "Running Type Promotion Tests..." << std::endl;

    // 1. Int8 + UInt8 -> Int16
    test_promotion(Dtype::Int8, Dtype::UInt8, Dtype::Int16, "Int8 + UInt8");
    test_promotion(Dtype::UInt8, Dtype::Int8, Dtype::Int16, "UInt8 + Int8");

    // 2. Float16 + BFloat16 -> Float32
    test_promotion(Dtype::Float16, Dtype::Bfloat16, Dtype::Float32, "Float16 + BFloat16");
    test_promotion(Dtype::Bfloat16, Dtype::Float16, Dtype::Float32, "BFloat16 + Float16");

    // 3. Complex Promotion
    test_promotion(Dtype::Complex32, Dtype::Float64, Dtype::Complex128, "Complex32 + Float64");
    test_promotion(Dtype::Complex32, Dtype::Float32, Dtype::Complex64, "Complex32 + Float32");
    test_promotion(Dtype::Complex32, Dtype::Float16, Dtype::Complex32, "Complex32 + Float16");
    test_promotion(Dtype::Complex32, Dtype::Bfloat16, Dtype::Complex64, "Complex32 + BFloat16");
    test_promotion(Dtype::Complex64, Dtype::Complex32, Dtype::Complex64, "Complex64 + Complex32");
    test_promotion(Dtype::Complex32, Dtype::Int64, Dtype::Complex64, "Complex32 + Int64");
    test_promotion(Dtype::Complex32, Dtype::Int32, Dtype::Complex64, "Complex32 + Int32");

    // 4. Standard Integer Behavior
    test_promotion(Dtype::Int8, Dtype::Int8, Dtype::Int8, "Int8 + Int8");
    test_promotion(Dtype::UInt8, Dtype::UInt8, Dtype::UInt8, "UInt8 + UInt8");
    test_promotion(Dtype::Int32, Dtype::Int8, Dtype::Int32, "Int32 + Int8");

    // 5. Standard Float Behavior
    test_promotion(Dtype::Float32, Dtype::Float16, Dtype::Float32, "Float32 + Float16");
    test_promotion(Dtype::Float64, Dtype::Float32, Dtype::Float64, "Float64 + Float32");
    test_promotion(Dtype::Int32, Dtype::Float64, Dtype::Float64, "Int32 + Float64");
    test_promotion(Dtype::Int64, Dtype::Float32, Dtype::Float64, "Int64 + Float32");

    std::cout << "All Type Promotion Tests Passed!" << std::endl;
    return 0;
}
