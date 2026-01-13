// sherpa-onnx/csrc/mtk/mtk-npu-executor.h
//
// Copyright (c)  2024  Xiaomi Corporation
// Copyright (c)  2024  MediaTek Inc.

#ifndef SHERPA_ONNX_CSRC_MTK_MTK_NPU_EXECUTOR_H_
#define SHERPA_ONNX_CSRC_MTK_MTK_NPU_EXECUTOR_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sherpa_onnx {

// Data type for tensor buffers
enum class MtkTensorDataType {
  kFloat32 = 1,
  kInt32 = 2,
  kUInt8 = 3,
  kFloat16 = 10,
  kInt8 = 9,
};

// Tensor buffer structure for input/output
struct MtkTensorBuffer {
  void* data;
  size_t bytes;
  MtkTensorDataType type;
};

// MTK NeuroPilot NPU Executor
// Handles loading and running DLA models on MTK NPU
class MtkNpuExecutor {
 public:
  explicit MtkNpuExecutor(const std::string& name);
  ~MtkNpuExecutor();

  // Initialize from DLA file path
  bool Initialize(const std::string& model_path,
                  const std::vector<std::vector<uint32_t>>& input_shapes,
                  const std::vector<std::vector<uint32_t>>& output_shapes,
                  int input_type,
                  int output_type,
                  const std::string& options = "");

  // Run inference with multiple inputs and outputs
  bool RunForMultipleInputsOutputs(const std::vector<MtkTensorBuffer>& inputs,
                                   const std::vector<MtkTensorBuffer>& outputs);

  // Get tensor sizes
  size_t GetInputTensorSize(size_t index) const;
  size_t GetOutputTensorSize(size_t index) const;

  // Check if initialized successfully
  bool IsInitialized() const { return is_initialized_; }

  // Get name
  const std::string& GetName() const { return name_; }

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
  std::string name_;
  bool is_initialized_ = false;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_MTK_MTK_NPU_EXECUTOR_H_
