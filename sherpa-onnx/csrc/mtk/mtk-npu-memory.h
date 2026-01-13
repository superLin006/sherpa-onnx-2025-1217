// sherpa-onnx/csrc/mtk/mtk-npu-memory.h
//
// Copyright (c)  2024  Xiaomi Corporation
// Copyright (c)  2024  MediaTek Inc.

#ifndef SHERPA_ONNX_CSRC_MTK_MTK_NPU_MEMORY_H_
#define SHERPA_ONNX_CSRC_MTK_MTK_NPU_MEMORY_H_

#include <cstddef>
#include <string>

namespace sherpa_onnx {

// Memory allocator for MTK NeuroPilot NPU
// This is a placeholder class - the current implementation uses
// regular heap memory with NON_ION_FD BufferAttribute.
// For ION/DMA-BUF memory optimization in the future, this class
// can be extended to manage NeuronMemory objects.
class MtkNpuMemory {
 public:
  enum class Kind {
    kHeap,  // Regular heap memory (current implementation)
    kIon,   // ION memory (for future optimization)
  };

  MtkNpuMemory(Kind kind, size_t size, const std::string& identifier);
  ~MtkNpuMemory();

  // Move constructor
  MtkNpuMemory(MtkNpuMemory&& other) noexcept;

  // Disable copy
  MtkNpuMemory(const MtkNpuMemory&) = delete;
  MtkNpuMemory& operator=(const MtkNpuMemory&) = delete;

  size_t GetSize() const { return size_; }
  void* GetAddr() const { return vaddr_; }
  bool IsAllocated() const { return is_allocated_; }

 private:
  bool Allocate(size_t size, const std::string& identifier);

  Kind kind_;
  size_t size_ = 0;
  std::string identifier_;
  bool is_allocated_ = false;
  void* vaddr_ = nullptr;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_MTK_MTK_NPU_MEMORY_H_
