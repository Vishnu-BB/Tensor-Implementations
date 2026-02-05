#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "TensorLib.h"

namespace fs = std::filesystem;

// static int getenv_int(const char* key, int def) {
//     const char* v = std::getenv(key);
//     return v ? std::atoi(v) : def;
// }

static std::vector<std::string> list_shards(const std::string& root,
                                            const std::string& split,
                                            const std::string& ext = ".bin") {
    std::vector<std::string> shards;
    if (!fs::exists(root)) return shards;
    for (const auto& e : fs::directory_iterator(root)) {
        if (!e.is_regular_file()) continue;
        auto p = e.path();
        std::string name = p.filename().string();
        if (p.extension() == ext && name.find(split) != std::string::npos) {
            shards.push_back(p.string());
        }
    }
    std::sort(shards.begin(), shards.end());
    return shards;
}

class UInt16ShardView {
public:
    UInt16ShardView() = default;
    ~UInt16ShardView() { close(); }

    void open(const std::string& path, size_t max_tokens) {
        close();
        path_ = path;

        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("failed to open: " + path);

        struct stat st {};
        if (fstat(fd_, &st) != 0) {
            ::close(fd_); fd_ = -1;
            throw std::runtime_error("failed to stat: " + path);
        }

        file_bytes_ = static_cast<size_t>(st.st_size);
        if (file_bytes_ % sizeof(uint16_t) != 0) {
            ::close(fd_); fd_ = -1;
            throw std::runtime_error("file size not divisible by 2 (uint16): " + path);
        }

        size_t total_tokens = file_bytes_ / 2;
        tokens_ = std::min(total_tokens, max_tokens);

        data_ = ::mmap(nullptr, file_bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (data_ == MAP_FAILED) {
            ::close(fd_); fd_ = -1; data_ = nullptr;
            throw std::runtime_error("mmap failed: " + path);
        }
    }

    void close() {
        if (data_) { ::munmap(data_, file_bytes_); data_ = nullptr; }
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
        file_bytes_ = 0; tokens_ = 0; path_.clear();
    }

    size_t size_tokens() const { return tokens_; }
    const std::string& path() const { return path_; }

    void read_block(size_t start, size_t count, std::vector<uint16_t>& out) const {
        out.resize(count);
        const uint16_t* p = reinterpret_cast<const uint16_t*>(data_);
        for (size_t i = 0; i < count; ++i) {
            // Wrap around the available tokens using modulo
            out[i] = p[(start + i) % tokens_];
        }
    }

private:
    std::string path_;
    int fd_ = -1;
    void* data_ = nullptr;
    size_t file_bytes_ = 0;
    size_t tokens_ = 0;
};

struct Batch {
    int B = 0, T = 0;
    OwnTensor::Tensor input;
    OwnTensor::Tensor target;
};

class DataLoaderLite {
public:
    DataLoaderLite(int B, int T, int rank, int world_size, const std::string& split, const std::string& data_root, size_t max_tokens_per_shard = 400000000)
        : B_(B), T_(T), rank_(rank), world_(world_size), split_(split), root_(data_root), max_tokens_(max_tokens_per_shard) {
        shards_ = list_shards(root_, split_, ".bin");
        if (shards_.empty()) throw std::runtime_error("no shards found");
        reset();
    }

    void reset() {
        current_shard_ = 0;
        shard_.open(shards_[current_shard_], max_tokens_);
        pos_ = static_cast<size_t>(B_) * static_cast<size_t>(T_) * static_cast<size_t>(rank_);
    }

    Batch next_batch() {
        const size_t BT = static_cast<size_t>(B_) * static_cast<size_t>(T_);
        if (pos_ + BT + 1 > shard_.size_tokens()) advance_shard();

        std::vector<uint16_t> buf;
        shard_.read_block(pos_, BT + 1, buf);

        Batch b;
        b.B = B_; b.T = T_;
        
        std::vector<uint16_t> x(BT), y(BT);
        for (size_t i = 0; i < BT; ++i) { x[i] = buf[i]; y[i] = buf[i + 1]; }

        OwnTensor::Device dev = OwnTensor::device::cuda_available() ? OwnTensor::Device::CUDA : OwnTensor::Device::CPU;
        
        b.input = OwnTensor::Tensor(OwnTensor::Shape{{B_, T_}}, {OwnTensor::Dtype::UInt16, OwnTensor::DeviceIndex(dev, rank_)});
        b.input.set_data(x);
        
        b.target = OwnTensor::Tensor(OwnTensor::Shape{{B_, T_}}, {OwnTensor::Dtype::UInt16, OwnTensor::DeviceIndex(dev, rank_)});
        b.target.set_data(y);

        pos_ += BT * static_cast<size_t>(world_);
        return b;
    }

private:
    void advance_shard() {
        current_shard_ = (current_shard_ + 1) % shards_.size();
        shard_.open(shards_[current_shard_], max_tokens_);
        pos_ = static_cast<size_t>(B_) * static_cast<size_t>(T_) * static_cast<size_t>(rank_);
    }

    int B_, T_, rank_, world_;
    std::string split_, root_;
    size_t max_tokens_, current_shard_ = 0, pos_ = 0;
    std::vector<std::string> shards_;
    UInt16ShardView shard_;
};


// #include <algorithm>
// #include <cstdint>
// #include <cstdlib>
// #include <filesystem>
// #include <iostream>
// #include <stdexcept>
// #include <string>
// #include <vector>
// #include <chrono>

// #include <cuda_runtime.h>

// #include <fcntl.h>
// #include <sys/mman.h>
// #include <sys/stat.h>
// #include <unistd.h>

// #include <mpi.h>

// #include "TensorLib.h"
// #include <cstring>


// namespace fs = std::filesystem;
// using steady_clock_t = std::chrono::steady_clock;

// static int getenv_int(const char* key, int def) {
//     const char* v = std::getenv(key);
//     return v ? std::atoi(v) : def;
// }

// static std::vector<std::string> list_shards(const std::string& root,
//                                             const std::string& split,
//                                             const std::string& ext = ".bin") {
//     std::vector<std::string> shards;
//     for (const auto& e : fs::directory_iterator(root)) {
//         if (!e.is_regular_file()) continue;
//         auto p = e.path();
//         std::string name = p.filename().string();
//         if (p.extension() == ext && name.find(split) != std::string::npos) {
//             shards.push_back(p.string());
//         }
//     }
//     std::sort(shards.begin(), shards.end());
//     return shards;
// }

// /* ======================= mmap shard ======================= */

// class UInt16ShardView {
// public:
//     ~UInt16ShardView() { close(); }

//     void open(const std::string& path, size_t max_tokens) {
//         close();
//         path_ = path;

//         fd_ = ::open(path.c_str(), O_RDONLY);
//         if (fd_ < 0) throw std::runtime_error("open failed: " + path);

//         struct stat st {};
//         if (fstat(fd_, &st) != 0)
//             throw std::runtime_error("stat failed: " + path);

//         file_bytes_ = st.st_size;
//         if (file_bytes_ % sizeof(uint16_t) != 0)
//             throw std::runtime_error("file not uint16 aligned");

//         size_t total_tokens = file_bytes_ / sizeof(uint16_t);
//         tokens_ = std::min(total_tokens, max_tokens);

//         data_ = ::mmap(nullptr, file_bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
//         if (data_ == MAP_FAILED)
//             throw std::runtime_error("mmap failed");
//     }

//     void close() {
//         if (data_) ::munmap(data_, file_bytes_);
//         if (fd_ >= 0) ::close(fd_);
//         data_ = nullptr;
//         fd_ = -1;
//         tokens_ = 0;
//         file_bytes_ = 0;
//     }

//     size_t size_tokens() const { return tokens_; }

//     void read_block(size_t start, size_t count,
//                     std::vector<uint16_t>& out) const {
//         out.resize(count);
//         const uint16_t* p = reinterpret_cast<const uint16_t*>(data_);
//         std::memcpy(out.data(), p + start, count * sizeof(uint16_t));
//     }

// private:
//     std::string path_;
//     int fd_ = -1;
//     void* data_ = nullptr;
//     size_t file_bytes_ = 0;
//     size_t tokens_ = 0;
// };

// /* ======================= batch ======================= */

// struct Batch {
//     int B = 0, T = 0;
//     std::vector<uint16_t> x, y;
//     OwnTensor::Tensor input;
//     OwnTensor::Tensor target;
// };

// struct BatchTiming {
//     double read_ms = 0.0;
//     double tensor_ms = 0.0;
// };

// /* ======================= dataloader ======================= */

// class DataLoaderLite {
// public:
//     DataLoaderLite(int B, int T,
//                    int rank, int world,
//                    const std::string& split,
//                    const std::string& root,
//                    size_t max_tokens = 400000000)
//         : B_(B), T_(T),
//           rank_(rank), world_(world),
//           split_(split), root_(root),
//           max_tokens_(max_tokens) {

//         shards_ = list_shards(root_, split_);
//         if (shards_.empty())
//             throw std::runtime_error("no shards found");

//         reset();
//     }

//     void reset() {
//         shard_idx_ = 0;
//         shard_.open(shards_[shard_idx_], max_tokens_);
//         pos_ = size_t(B_) * T_ * rank_;
//     }

//     Batch next_batch(BatchTiming* timing = nullptr) {
//         const size_t BT = size_t(B_) * T_;
//         const size_t need = BT + 1;

//         if (pos_ + need > shard_.size_tokens())
//             advance_shard();

//         auto t0 = steady_clock_t::now();

//         std::vector<uint16_t> buf;
//         shard_.read_block(pos_, need, buf);

//         auto t1 = steady_clock_t::now();

//         Batch b;
//         b.B = B_;
//         b.T = T_;
//         b.x.resize(BT);
//         b.y.resize(BT);

//         for (size_t i = 0; i < BT; ++i) {
//             b.x[i] = buf[i];
//             b.y[i] = buf[i + 1];
//         }

//         b.input  = OwnTensor::Tensor({{B_, T_}},
//                         {OwnTensor::Dtype::UInt16, OwnTensor::Device::CPU});
//         b.target = OwnTensor::Tensor({{B_, T_}},
//                         {OwnTensor::Dtype::UInt16, OwnTensor::Device::CPU});

//         b.input.set_data(b.x);
//         b.target.set_data(b.y);

//         auto t2 = steady_clock_t::now();

//         if (timing) {
//             timing->read_ms =
//                 std::chrono::duration<double, std::milli>(t1 - t0).count();
//             timing->tensor_ms =
//                 std::chrono::duration<double, std::milli>(t2 - t1).count();
//         }

//         pos_ += BT * world_;
//         if (pos_ + (BT * world_ + 1) > shard_.size_tokens())
//             advance_shard();

//         return b;
//     }

// private:
//     void advance_shard() {
//         shard_idx_ = (shard_idx_ + 1) % shards_.size();
//         shard_.open(shards_[shard_idx_], max_tokens_);
//         pos_ = size_t(B_) * T_ * rank_;
//     }

//     int B_, T_;
//     int rank_, world_;
//     std::string split_, root_;
//     size_t max_tokens_;

//     std::vector<std::string> shards_;
//     size_t shard_idx_ = 0;
//     size_t pos_ = 0;

//     UInt16ShardView shard_;
// };

/* ======================= main benchmark ======================= */

// int main(int argc, char** argv) {
//     MPI_Init(&argc, &argv);

//     int rank, world;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &world);

//     cudaSetDevice(rank);

//     const std::string data_root =
//         "/home/blu-bridge015/Desktop/Data Parallel/DataLoader/";
//     const int B = 16;
//     const int T = 1048576;

//     const int WARMUP = 5;
//     const int ITERS  = 50;

//     DataLoaderLite loader(B, T, rank, world, "train", data_root);

//     MPI_Barrier(MPI_COMM_WORLD);

//     /* warmup */
//     for (int i = 0; i < WARMUP; ++i)
//         loader.next_batch();

//     MPI_Barrier(MPI_COMM_WORLD);

//     double total_ms = 0.0;
//     double read_ms = 0.0;
//     double tensor_ms = 0.0;

//     for (int i = 0; i < ITERS; ++i) {
//         BatchTiming t;
//         auto t0 = steady_clock_t::now();
//         Batch b = loader.next_batch(&t);
//         b.input.to(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank));
//         b.target.to(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank));
//         auto t1 = steady_clock_t::now();

//         total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
//         read_ms += t.read_ms;
//         tensor_ms += t.tensor_ms;
//     }

//     double avg_ms = total_ms / ITERS;
//     double tokens_per_sec =
//         (double(B) * T) / (avg_ms / 1000.0);

//     double global_tokens_sec = 0.0;
//     double max_batch_ms = 0.0;

//     MPI_Reduce(&tokens_per_sec, &global_tokens_sec,
//                1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
//     MPI_Reduce(&avg_ms, &max_batch_ms,
//                1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

//     if (rank == 0) {
//         std::cout << "\n=== DataLoader Benchmark ===\n";
//         std::cout << "World size: " << world << "\n";
//         std::cout << "B=" << B << " T=" << T << "\n";
//         std::cout << "Max avg batch time: " << max_batch_ms << " ms\n";
//         std::cout << "Global throughput: "
//                   << global_tokens_sec / 1e6
//                   << " M tokens/s\n";
//     }

//     MPI_Finalize();
//     return 0;
// }
