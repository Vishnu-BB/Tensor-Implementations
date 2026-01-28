#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace OwnTensor {

enum class AllocEvent { ALLOC, FREE };

// Lifetime hints for allocations - helps identify unexpected leaks
enum class AllocLifetime {
  UNKNOWN,    // Default
  TEMPORARY,  // Should be freed quickly (intermediates, gradients in forward)
  ACTIVATION, // Activations for backward pass
  WEIGHT,     // Model weights - expected to live long
  BUFFER      // Workspace buffers
};

inline const char *lifetime_to_str(AllocLifetime lt) {
  switch (lt) {
  case AllocLifetime::TEMPORARY:
    return "TEMPORARY";
  case AllocLifetime::ACTIVATION:
    return "ACTIVATION";
  case AllocLifetime::WEIGHT:
    return "WEIGHT";
  case AllocLifetime::BUFFER:
    return "BUFFER";
  default:
    return "UNKNOWN";
  }
}

struct AllocationRecord {
  uint64_t alloc_id; // Unique monotonic ID per allocation
  std::string name_id;
  size_t bytes;
  int device;
  uint64_t alloc_time_ns; // When allocated (for duration tracking)
  AllocLifetime lifetime; // Expected lifetime hint
};

struct LeakInfo {
  uint64_t alloc_id;
  std::string name;
  size_t bytes;
  int device;
  uint64_t age_ns; // How long it's been alive
  AllocLifetime lifetime;
};

class AllocationTracker {
public:
  static AllocationTracker &instance();

  void init(const char *csv_path);
  void shutdown();

  // Thread-local naming with lifetime hints
  static void set_thread_name(const std::string &name,
                              AllocLifetime lifetime = AllocLifetime::UNKNOWN) {
    current_name() = name;
    current_lifetime() = lifetime;
  }
  static AllocLifetime get_current_lifetime() {
    return current_lifetime();
  }
  static void clear_thread_name() {
    current_name() = "unknown";
    current_lifetime() = AllocLifetime::UNKNOWN;
  }

  void on_alloc(void *ptr, size_t bytes, int device);
  void on_free(void *ptr, int device);

  // === New diagnostic methods ===

  // Get all currently live allocations that haven't been freed
  std::vector<LeakInfo> report_leaks() const;

  // Print leak report to stderr (convenience method)
  void print_leak_report() const;

  // Get memory statistics
  size_t get_current_allocated(int device = -1) const; // -1 = all devices
  size_t get_peak_allocated(int device = -1) const;
  size_t get_total_allocations() const { return alloc_counter_.load(); }

  // Reset peak tracking (useful between training epochs)
  void reset_peak();

  // Enable/disable live console logging
  void enable_console_logging(bool enable) { console_logging_ = enable; }

private:
  AllocationTracker() = default;

  static std::string &current_name() {
    thread_local std::string name = "unknown";
    return name;
  }
  static AllocLifetime &current_lifetime() {
    thread_local AllocLifetime lt = AllocLifetime::UNKNOWN;
    return lt;
  }

  mutable std::mutex mtx_;
  std::unordered_map<void *, AllocationRecord> live_allocs_;
  std::ofstream csv_;
  bool initialized_ = false;

  // Allocation ID counter (monotonically increasing)
  std::atomic<uint64_t> alloc_counter_{0};

  // Memory tracking per device (-1 = CPU, 0+ = GPU devices)
  std::unordered_map<int, size_t> current_bytes_; // Currently allocated
  std::unordered_map<int, size_t> peak_bytes_;    // Peak allocation
  bool console_logging_ = false;
};

// RAII helper for scoped naming
class ScopedAllocName {
public:
  ScopedAllocName(const std::string &name,
                  AllocLifetime lifetime = AllocLifetime::UNKNOWN) {
    AllocationTracker::set_thread_name(name, lifetime);
  }
  ~ScopedAllocName() { AllocationTracker::clear_thread_name(); }

  // Non-copyable
  ScopedAllocName(const ScopedAllocName &) = delete;
  ScopedAllocName &operator=(const ScopedAllocName &) = delete;
};

} // namespace OwnTensor