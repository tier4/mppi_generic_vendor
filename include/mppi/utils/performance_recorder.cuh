// Copyright 2026 TIER IV, Inc.
// SPDX-License-Identifier: Apache-2.0

#ifndef MPPI_UTILS_PERFORMANCE_RECORDER_CUH_
#define MPPI_UTILS_PERFORMANCE_RECORDER_CUH_

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace mppi
{
namespace instrumentation
{

struct PerformancePhase
{
  std::string name;
  std::size_t calls = 0;
  std::size_t gpu_calls = 0;
  double host_ms = 0.0;
  double gpu_ms = 0.0;
};

/**
 * Optional host-thread recorder. Allocate its event pool before measurement, activate it with
 * ScopedPerformanceRecording, then collect after the caller's completion synchronization.
 * Ranges can nest: host durations are inclusive and must not be summed. GPU durations are stream
 * elapsed times (including any scheduling/launch gaps), not utilization or exclusive kernel time.
 */
class PerformanceRecorder
{
public:
  using Clock = std::chrono::steady_clock;
  static constexpr std::size_t no_slot = static_cast<std::size_t>(-1);

  explicit PerformanceRecorder(std::size_t range_capacity, std::size_t gpu_capacity)
    : ranges_(range_capacity), events_(gpu_capacity)
  {
    for (auto& pair : events_)
    {
      check(cudaEventCreate(&pair.start));
      check(cudaEventCreate(&pair.stop));
    }
  }

  PerformanceRecorder(const PerformanceRecorder&) = delete;
  PerformanceRecorder& operator=(const PerformanceRecorder&) = delete;

  void reset() noexcept
  {
    used_ = 0;
    gpu_used_ = 0;
    overflow_ = false;
    error_ = cudaSuccess;
  }

  // Name must remain alive until collect(); NVTX call sites use string literals.
  std::size_t begin(const char* name, bool gpu, cudaStream_t stream) noexcept
  {
    if (used_ == ranges_.size() || (gpu && gpu_used_ == events_.size()))
    {
      overflow_ = true;
      return no_slot;
    }
    const std::size_t index = used_++;
    auto& range = ranges_[index];
    range.name = name;
    range.event = gpu ? gpu_used_++ : no_slot;
    range.stream = stream;
    range.finished = false;
    range.start = Clock::now();
    if (gpu)
      remember(cudaEventRecord(events_[range.event].start, stream));
    return index;
  }

  void end(std::size_t index) noexcept
  {
    if (index == no_slot)
      return;
    auto& range = ranges_[index];
    if (range.event != no_slot)
      remember(cudaEventRecord(events_[range.event].stop, range.stream));
    range.host_ms = std::chrono::duration<double, std::milli>(Clock::now() - range.start).count();
    range.finished = true;
  }

  /** No synchronization or CUDA allocation here. Incomplete GPU work is an error, not a zero. */
  std::vector<PerformancePhase> collect() const
  {
    if (overflow_)
      throw std::runtime_error("MPPI performance recorder capacity exceeded");
    check(error_);
    std::map<std::string, PerformancePhase> phases;
    for (std::size_t i = 0; i < used_; ++i)
    {
      const auto& range = ranges_[i];
      if (!range.finished)
        throw std::runtime_error("MPPI performance range is still active");
      auto& phase = phases[range.name];
      phase.name = range.name;
      ++phase.calls;
      phase.host_ms += range.host_ms;
      if (range.event != no_slot)
      {
        const auto& pair = events_[range.event];
        check(cudaEventQuery(pair.stop));
        float milliseconds = 0.0F;
        check(cudaEventElapsedTime(&milliseconds, pair.start, pair.stop));
        phase.gpu_ms += milliseconds;
        ++phase.gpu_calls;
      }
    }
    std::vector<PerformancePhase> result;
    result.reserve(phases.size());
    for (const auto& phase : phases)
      result.push_back(phase.second);
    return result;
  }

private:
  struct EventPair
  {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    EventPair() = default;
    EventPair(const EventPair&) = delete;
    EventPair& operator=(const EventPair&) = delete;
    ~EventPair() noexcept
    {
      if (stop)
        (void)cudaEventDestroy(stop);
      if (start)
        (void)cudaEventDestroy(start);
    }
  };
  struct Range
  {
    const char* name = nullptr;
    std::size_t event = no_slot;
    cudaStream_t stream = nullptr;
    Clock::time_point start;
    double host_ms = 0.0;
    bool finished = false;
  };

  static void check(cudaError_t status)
  {
    if (status != cudaSuccess)
      throw std::runtime_error(std::string("MPPI performance timing: ") + cudaGetErrorString(status));
  }
  void remember(cudaError_t status) noexcept
  {
    if (error_ == cudaSuccess)
      error_ = status;
  }

  std::vector<Range> ranges_;
  std::vector<EventPair> events_;
  std::size_t used_ = 0;
  std::size_t gpu_used_ = 0;
  bool overflow_ = false;
  cudaError_t error_ = cudaSuccess;
};

inline PerformanceRecorder*& activePerformanceRecorder() noexcept
{
  static thread_local PerformanceRecorder* recorder = nullptr;
  return recorder;
}

/** Restores the previous thread context even when optimization throws. */
class ScopedPerformanceRecording
{
public:
  explicit ScopedPerformanceRecording(PerformanceRecorder* recorder) noexcept : previous_(activePerformanceRecorder())
  {
    if (recorder)
      recorder->reset();
    activePerformanceRecorder() = recorder;
  }
  ~ScopedPerformanceRecording() noexcept
  {
    activePerformanceRecorder() = previous_;
  }
  ScopedPerformanceRecording(const ScopedPerformanceRecording&) = delete;
  ScopedPerformanceRecording& operator=(const ScopedPerformanceRecording&) = delete;

private:
  PerformanceRecorder* previous_;
};

/** No clock reads, event operations or allocations when recording is disabled. */
class ScopedPerformanceRange
{
public:
  ScopedPerformanceRange(const char* name, bool gpu, cudaStream_t stream) noexcept
    : recorder_(activePerformanceRecorder())
    , index_(recorder_ ? recorder_->begin(name, gpu, stream) : PerformanceRecorder::no_slot)
  {
  }
  ~ScopedPerformanceRange() noexcept
  {
    if (recorder_)
      recorder_->end(index_);
  }
  ScopedPerformanceRange(const ScopedPerformanceRange&) = delete;
  ScopedPerformanceRange& operator=(const ScopedPerformanceRange&) = delete;

private:
  PerformanceRecorder* recorder_;
  std::size_t index_;
};

}  // namespace instrumentation
}  // namespace mppi

#endif  // MPPI_UTILS_PERFORMANCE_RECORDER_CUH_
