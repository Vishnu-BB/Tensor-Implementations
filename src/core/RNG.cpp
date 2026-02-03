#include "core/RNG.h"
#include <sstream>
#include <stdexcept>
#include <iostream>

namespace OwnTensor {

thread_local std::unique_ptr<std::mt19937> RNG::cpu_gen_ = nullptr;
#ifdef WITH_CUDA
thread_local curandGenerator_t RNG::gpu_gen_;
thread_local unsigned long long RNG::gpu_seed_ = 1234ULL;
thread_local unsigned long long RNG::gpu_offset_ = 0ULL;
thread_local bool RNG::gpu_gen_initialized_ = false;
#endif

std::mt19937& RNG::get_cpu_generator() {
    if (!cpu_gen_) {
        cpu_gen_ = std::make_unique<std::mt19937>(5489u);  // mt19937 default seed
    }
    return *cpu_gen_;
}
#ifdef WITH_CUDA
curandGenerator_t RNG::get_gpu_generator() {
    if (!gpu_gen_initialized_) {
        curandCreateGenerator(&gpu_gen_, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gpu_gen_, gpu_seed_);
        curandSetGeneratorOffset(gpu_gen_, gpu_offset_);
        gpu_gen_initialized_ = true;
    }
    return gpu_gen_;
}

void RNG::increment_gpu_offset(size_t count) {
    gpu_offset_ += count;
}
#endif

RNGState RNG::get_state() {
    RNGState state;
    
    // Capture CPU state
    auto& gen = get_cpu_generator();
    std::stringstream ss;
    ss << gen;
    state.cpu_state = ss.str();

#ifdef WITH_CUDA
    state.gpu_seed = gpu_seed_;
    state.gpu_offset = gpu_offset_;
    // Note: curand doesn't easily expose internal state of the generator.
    // We rely on seed and offset for recomputation.
#endif
    return state;
}

void RNG::set_state(const RNGState& state) {
    // Restore CPU state
    auto& gen = get_cpu_generator();
    std::stringstream ss(state.cpu_state);
    ss >> gen;

#ifdef WITH_CUDA
    gpu_seed_ = state.gpu_seed;
    gpu_offset_ = state.gpu_offset;
    if (gpu_gen_initialized_) {
        curandSetPseudoRandomGeneratorSeed(gpu_gen_, gpu_seed_);
        curandSetGeneratorOffset(gpu_gen_, gpu_offset_);
    }
#endif
}

void RNG::set_seed(unsigned long seed) {
    get_cpu_generator().seed(seed);
#ifdef WITH_CUDA
    gpu_seed_ = static_cast<unsigned long long>(seed);
    gpu_offset_ = 0ULL;
    if (gpu_gen_initialized_) {
        curandSetPseudoRandomGeneratorSeed(gpu_gen_, gpu_seed_);
        curandSetGeneratorOffset(gpu_gen_, gpu_offset_);
    }
#endif
}

} // namespace OwnTensor