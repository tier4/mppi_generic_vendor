#pragma once

#ifndef MPPI_UTILS_NVTX_CUH_
#define MPPI_UTILS_NVTX_CUH_

#include <nvtx3/nvToolsExt.h>
#include <cuda_runtime_api.h>
#include <mppi/utils/performance_recorder.cuh>

#include <cstdint>

namespace mppi
{
namespace instrumentation
{

/** Stable colors used to distinguish MPPI phases in Nsight Systems. */
enum class NvtxColor : std::uint32_t
{
  DATA_TRANSFER = 0xFF4E79A7U,
  MAP_GENERATION = 0xFF59A14FU,
  SAMPLING = 0xFFF28E2BU,
  ROLLOUT = 0xFFE15759U,
  STATISTICS = 0xFFB07AA1U,
  REDUCTION = 0xFF76B7B2U,
  DEBUG = 0xFF9C755FU,
};

/** Exception-safe host-thread NVTX range with no CUDA synchronization. */
class ScopedNvtxRange
{
public:
  __host__ explicit ScopedNvtxRange(const char* name, const NvtxColor color) noexcept
    : performance_(name, false, nullptr)
  {
    push(name, color);
  }

  /** Also time queued device work on this exact stream when a recorder is active. */
  __host__ ScopedNvtxRange(const char* name, const NvtxColor color, cudaStream_t stream) noexcept
    : performance_(name, true, stream)
  {
    push(name, color);
  }

private:
  __host__ static void push(const char* name, const NvtxColor color) noexcept
  {
    nvtxEventAttributes_t attributes{};
    attributes.version = NVTX_VERSION;
    attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attributes.colorType = NVTX_COLOR_ARGB;
    attributes.color = static_cast<std::uint32_t>(color);
    attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attributes.message.ascii = name;
    nvtxRangePushEx(&attributes);
  }

public:
  __host__ ~ScopedNvtxRange() noexcept
  {
    nvtxRangePop();
  }

  ScopedNvtxRange(const ScopedNvtxRange&) = delete;
  ScopedNvtxRange& operator=(const ScopedNvtxRange&) = delete;
  ScopedNvtxRange(ScopedNvtxRange&&) = delete;
  ScopedNvtxRange& operator=(ScopedNvtxRange&&) = delete;

private:
  ScopedPerformanceRange performance_;
};

}  // namespace instrumentation
}  // namespace mppi

#endif  // MPPI_UTILS_NVTX_CUH_
