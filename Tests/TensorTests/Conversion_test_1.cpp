#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include "dtype/Types.h"

using namespace OwnTensor;
using namespace OwnTensor::detail;

void print_result(const char* name, float val, float expected) {
    bool match = (std::isnan(val) && std::isnan(expected)) || 
                 (std::isinf(val) && std::isinf(expected) && (val > 0) == (expected > 0)) ||
                 (std::abs(val - expected) < 1e-5);
    
    std::cout << "[" << (match ? "PASS" : "FAIL") << "] " << name 
              << ": Got " << val << ", Expected " << expected << std::endl;
}

int main() {
    std::cout << "=== Verifying FP16 Conversions (Optimized) ===" << std::endl;
    
    // Test 1: Normal Value (1.0)
    // FP16 1.0 = 0x3C00
    float f1 = float16_to_float(0x3C00);
    print_result("FP16 1.0", f1, 1.0f);

    // Test 2: Zero
    float f2 = float16_to_float(0x0000);
    print_result("FP16 0.0", f2, 0.0f);

    // Test 3: Denormal (Smallest positive)
    // FP16 min denorm = 0x0001 = 2^-24 ≈ 5.96e-8
    float f3 = float16_to_float(0x0001);
    print_result("FP16 Min Denorm", f3, 5.96046e-8f);

    // Test 4: Infinity
    // FP16 Inf = 0x7C00
    float f4 = float16_to_float(0x7C00);
    print_result("FP16 Inf", f4, std::numeric_limits<float>::infinity());

    // Test 5: NaN
    // FP16 NaN = 0x7E00
    float f5 = float16_to_float(0x7E00);
    print_result("FP16 NaN", f5, std::numeric_limits<float>::quiet_NaN());


    std::cout << "\n=== Verifying FP8 E4M3FN Conversions (Simplified) ===" << std::endl;
    
    // Test 6: 1.0
    // E4M3 1.0 = 0x38 (0 0111 000) -> Exp=7 (bias 7) -> 2^0 * 1.0
    float f6 = e4m3fn_to_float(0x38);
    print_result("E4M3 1.0", f6, 1.0f);

    // Test 7: Max Value (448)
    // E4M3 Max = 0x7E (0 1111 110)
    float f7 = e4m3fn_to_float(0x7E);
    print_result("E4M3 Max", f7, 448.0f);

    // Test 8: NaN
    // E4M3 NaN = 0x7F (0 1111 111)
    float f8 = e4m3fn_to_float(0x7F);
    print_result("E4M3 NaN", f8, std::numeric_limits<float>::quiet_NaN());


    std::cout << "\n=== Verifying FP8 E5M2 Conversions (Simplified) ===" << std::endl;

    // Test 9: 1.0
    // E5M2 1.0 = 0x3C (0 01111 00) -> Exp=15 (bias 15) -> 2^0 * 1.0
    float f9 = e5m2_to_float(0x3C);
    print_result("E5M2 1.0", f9, 1.0f);

    // Test 10: Infinity
    // E5M2 Inf = 0x7C (0 11111 00)
    float f10 = e5m2_to_float(0x7C);
    print_result("E5M2 Inf", f10, std::numeric_limits<float>::infinity());

    return 0;
}
