#include "TensorLib.h"
#include <iostream>

using namespace OwnTensor;

int main()
{
    Tensor a = Tensor::rand({{25*1024*1024}}, {Dtype::Float32, DeviceIndex(Device::CPU, 0), false}, 42, -1.f, 2.f);
    // a.display();
    // std::cout << "\n";
    // std::vector<Tensor> shards = a.make_shards(3, false);

    // int i = 0;
    // for (Tensor shard : shards)
    // {
    //     std::cout << "\nShard " << i << ":\n";
    //     shard.display();
    //     ++i;
    // }

    {
    a.display();
    std::cout << "Address of A: " << a.data() << "\n";
    std::cin.get();
    std::cout << "\n=====================================\n";
    auto b = a.to_cuda(0);
    b.display();
    std::cout << "Address of A: " << a.data() << "\n";
    std::cin.get();
    std::cout << "\n=====================================\n";
    auto c = b.to_cpu();
    // b.impl_->storage().data_ptr() = nullptr;
    c.display();
    std::cout << "Address of A: " << a.data() << "\n";
    std::cin.get();
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
    // // If there's a leak, nvidia-smi would increase by 4GB EVERY iteration.
    // }

}
}
