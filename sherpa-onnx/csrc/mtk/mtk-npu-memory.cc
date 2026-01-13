// sherpa-onnx/csrc/mtk/mtk-npu-memory.cc
//
// Copyright (c)  2024  Xiaomi Corporation
// Copyright (c)  2024  MediaTek Inc.

#include "sherpa-onnx/csrc/mtk/mtk-npu-memory.h"

#include <cstdlib>
#include <cstring>

#include "sherpa-onnx/csrc/mtk/macros.h"

namespace sherpa_onnx {

MtkNpuMemory::MtkNpuMemory(Kind kind, size_t size, const std::string& identifier)
    : kind_(kind), identifier_(identifier) {
  is_allocated_ = Allocate(size, identifier);
}

MtkNpuMemory::~MtkNpuMemory() {
  if (is_allocated_ && vaddr_ != nullptr) {
    std::free(vaddr_);
    vaddr_ = nullptr;
  }
}

MtkNpuMemory::MtkNpuMemory(MtkNpuMemory&& other) noexcept
    : kind_(other.kind_),
      size_(other.size_),
      identifier_(std::move(other.identifier_)),
      is_allocated_(other.is_allocated_),
      vaddr_(other.vaddr_) {
  other.vaddr_ = nullptr;
  other.is_allocated_ = false;
}

bool MtkNpuMemory::Allocate(size_t size, const std::string& identifier) {
  // For now, we just use regular heap memory
  // In the future, ION/DMA-BUF memory can be used for better performance
  vaddr_ = std::malloc(size);
  if (vaddr_ == nullptr) {
    SHERPA_ONNX_LOGE("Failed to allocate %zu bytes for %s", size, identifier.c_str());
    return false;
  }

  std::memset(vaddr_, 0, size);
  size_ = size;

  SHERPA_ONNX_LOGI("Allocated %zu bytes for %s", size, identifier.c_str());
  return true;
}

}  // namespace sherpa_onnx
