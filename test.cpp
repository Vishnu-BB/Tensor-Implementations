#include "TensorLib.h"
#include <iostream>

using namespace OwnTensor;

int main()
{
    Tensor a = Tensor::rand({{3,3}}, {Dtype::Float32, DeviceIndex(Device::CPU, 0), false}, 42, -1.f, 2.f);
    a.display();
    std::cout << "\n";
    std::vector<Tensor> shards = a.make_shards(3, true);

    int i = 0;
    for (Tensor shard : shards)
    {
        std::cout << "\nShard " << i << ":\n";
        shard.display();
        ++i;
    }

    // {
    // a.display();
    // std::cout << "Address of A: " << a.data() << "\n";
    // std::cin.get();
    // std::cout << "\n=====================================\n";
    // a.to_cuda_(0);
    // a.display();
    // std::cout << "Address of A: " << a.data() << "\n";
    // std::cin.get();
    // std::cout << "\n=====================================\n";
    // a.to_cpu_();
    // a.display();
    // std::cout << "Address of A: " << a.data() << "\n";
    // std::cin.get();
    // for (int i = 0; i < 1; ++i) {
    // std::cout << "Address of a: " << a.data() << "\n";
    // std::cout << "Iteration " << i << " - Moving to GPU\n";
    // a.to_cuda_(); 
    // std::cout << "Address of a: " << a.data() << "\n";
    // // sleep(10);
    // // Check nvidia-smi here: it should be ~4GB + Context  
    // std::cout << "Iteration " << i << " - Moving to CPU\n";
    // a.to_cpu_();
    // std::cout << "Address of a: " << a.data() << "\n\n";
    // sleep(10);
    // If there's a leak, nvidia-smi would increase by 4GB EVERY iteration.
    // }

}
