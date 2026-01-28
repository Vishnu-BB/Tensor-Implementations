#include "TensorLib.h"
#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>

using namespace OwnTensor;

void print_time(const std::string& label) {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::cout << "[" << std::put_time(std::localtime(&time), "%H:%M:%S") << "] " << label << "\n";
}

int main()
{
    print_time("Starting test - creating CPU tensor (100MB)");
    Tensor a = Tensor::rand({{25*1024*1024}}, {Dtype::Float32, DeviceIndex(Device::CPU, 0), false}, 42, -1.f, 2.f);

    {
    std::cout << "\n=== STAGE 1: CPU Tensor Created ===\n";
    a.display();
    std::cout << "Address of A (CPU): " << a.data() << "\n";
    print_time("Check nvidia-smi now - GPU memory should be minimal");
    std::cout << "Press Enter to continue...\n";
    std::cin.get();
    
    std::cout << "\n=== STAGE 2: Moving to CUDA ===\n";
    print_time("Transferring to GPU...");
    a.to_cuda_(0);
    print_time("Transfer complete");
    a.display();
    std::cout << "Address of A (GPU): " << a.data() << "\n";
    print_time("Check nvidia-smi now - GPU memory should show ~100MB allocation");
    std::cout << "Press Enter to continue...\n";
    std::cin.get();
    
    std::cout << "\n=== STAGE 3: Moving back to CPU ===\n";
    print_time("Transferring back to CPU...");
    a.to_cpu_();
    print_time("Transfer complete");
    a.display();
    std::cout << "Address of A (CPU): " << a.data() << "\n";
    print_time("Check nvidia-smi now - GPU memory should be freed (async free completed)");
    std::cout << "Press Enter to continue...\n";
    std::cin.get();
    
    std::cout << "\n=== STAGE 4: Exiting scope ===\n";
    print_time("About to exit scope - b and c will be destroyed");
}
    print_time("Scope exited - GPU tensor b should be freed");
    std::cout << "Press Enter to exit...\n";
    std::cin.get();
    
    print_time("Test complete");
    return 0;
}
