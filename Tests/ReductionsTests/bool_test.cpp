#include <iostream>
#include "TensorLib.h"
using namespace OwnTensor;
using namespace std;
// #include <iostream>
int main(){

 cout<<"hi"<<endl;
Tensor T({{3,2}},Dtype::UInt16,Device::CUDA);
//std::vector<bool> data = {false,false,false,false,true,false,true,false,true};
// cout<<"hi"<<endl;
std::vector<uint16_t> data = {100, 79, 60, 90, 10, 0};
// cout<<"ji"<<endl;
//std::vector data = {complex32_t(100.0,5.0),complex32_t(100,5),complex32_t(100,5),complex32_t(100,5),complex32_t(100,5),complex32_t(100,5)};
T.set_data(data);
cout << "\n\n\n" << endl;
T.to_cpu().display();
Tensor T1({{3,2}},Dtype::UInt16,Device::CUDA);
std::vector<uint16_t> data1 = {100, 79, 60, 90, 10, 0};
T1.set_data(data1);
T1.to_cpu().display();
 Tensor T2=T+T1;
T2.to_cpu().display();
//  Tensor T1({{3,2}},Dtype::Float8_E4M3FN,Device::CUDA);
//  std::vector<float8_e4m3fn_t> data1 = {100.0, 79.0, 60.0, 90.0, 10.0, 0.0};
// // T1.fill(complex32_t(40,5));
// T1.set_data(data1);
//  Tensor res = T * T1 ;
//  res.to_cpu().display();

// Tensor res = reduce_max(T);
// res.display(std::cout, 4);
// Tensor Honey = reduce_all(T);
// Tensor Bunty = reduce_any(T);
// // Tensor x=Honey.to_cpu();
// // Tensor y=Bunty.to_cpu();
// Honey.to_cpu().display(cout,2);
// Bunty.to_cpu().display(cout,2);
}