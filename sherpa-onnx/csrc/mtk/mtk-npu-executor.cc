// sherpa-onnx/csrc/mtk/mtk-npu-executor.cc
//
// Copyright (c)  2024  Xiaomi Corporation
// Copyright (c)  2024  MediaTek Inc.

#include "sherpa-onnx/csrc/mtk/mtk-npu-executor.h"

#include <cstring>
#include <vector>

#include "sherpa-onnx/csrc/log.h"
#include "sherpa-onnx/csrc/mtk/macros.h"
#include "sherpa-onnx/csrc/mtk/neuron/NeuronRuntimeLibrary.h"
#include "sherpa-onnx/csrc/mtk/neuron/api/Types.h"

// BufferAttribute and NON_ION_FD are defined in Types.h (C-style)

namespace sherpa_onnx {

class MtkNpuExecutor::Impl {
 public:
  Impl() = default;

  ~Impl() {
    if (runtime_ != nullptr && neuron_lib_) {
      neuron_lib_->Release(runtime_);
      runtime_ = nullptr;
    }
  }

  bool Initialize(const std::string& model_path,
                  const std::vector<std::vector<uint32_t>>& input_shapes,
                  const std::vector<std::vector<uint32_t>>& output_shapes,
                  int input_type,
                  int output_type,
                  const std::string& options) {
    model_path_ = model_path;
    input_shapes_ = input_shapes;
    output_shapes_ = output_shapes;

    SHERPA_ONNX_LOGI("MTK NPU Executor initializing from: %s", model_path.c_str());

    // Initialize NeuronRuntimeLibrary (dynamically loads libneuron_runtime.so)
    neuron_lib_ = std::make_unique<mtk::neuropilot::NeuronRuntimeLibrary>();

    // Create runtime with options
    int ret;
    if (!options.empty()) {
      ret = neuron_lib_->CreateWithOptions(options.c_str(), nullptr, &runtime_);
    } else {
      ret = neuron_lib_->Create(nullptr, &runtime_);
    }

    if (ret != 0 || runtime_ == nullptr) {
      SHERPA_ONNX_LOGE("Failed to create Neuron runtime: %d", ret);
      return false;
    }

    // Load DLA model
    ret = neuron_lib_->LoadNetworkFromFile(runtime_, model_path.c_str());
    if (ret != 0) {
      SHERPA_ONNX_LOGE("Failed to load DLA file: %s, error: %d", model_path.c_str(), ret);
      return false;
    }

    // Pre-allocate buffers
    AllocateBuffers();

    SHERPA_ONNX_LOGI("MTK NPU Executor initialized successfully");
    return true;
  }

  bool Initialize(const void* model_buffer, size_t model_size,
                  const std::vector<std::vector<uint32_t>>& input_shapes,
                  const std::vector<std::vector<uint32_t>>& output_shapes,
                  int input_type,
                  int output_type,
                  const std::string& options) {
    input_shapes_ = input_shapes;
    output_shapes_ = output_shapes;

    SHERPA_ONNX_LOGI("MTK NPU Executor initializing from memory buffer (size: %zu bytes)", model_size);

    // Initialize NeuronRuntimeLibrary (dynamically loads libneuron_runtime.so)
    neuron_lib_ = std::make_unique<mtk::neuropilot::NeuronRuntimeLibrary>();

    // Create runtime with options
    int ret;
    if (!options.empty()) {
      ret = neuron_lib_->CreateWithOptions(options.c_str(), nullptr, &runtime_);
    } else {
      ret = neuron_lib_->Create(nullptr, &runtime_);
    }

    if (ret != 0 || runtime_ == nullptr) {
      SHERPA_ONNX_LOGE("Failed to create Neuron runtime: %d", ret);
      return false;
    }

    // Load DLA model from memory buffer
    ret = neuron_lib_->LoadNetworkFromBuffer(runtime_, model_buffer, model_size);
    if (ret != 0) {
      SHERPA_ONNX_LOGE("Failed to load DLA from memory buffer, error: %d", ret);
      return false;
    }

    // Pre-allocate buffers
    AllocateBuffers();

    SHERPA_ONNX_LOGI("MTK NPU Executor initialized successfully from memory");
    return true;
  }

  bool RunForMultipleInputsOutputs(const std::vector<MtkTensorBuffer>& inputs,
                                   const std::vector<MtkTensorBuffer>& outputs) {
    if (!neuron_lib_ || !runtime_) {
      SHERPA_ONNX_LOGE("Neuron runtime not initialized");
      return false;
    }

    // BufferAttribute for non-ION memory (regular heap memory)
    BufferAttribute buf_attr{NON_ION_FD};

    // Set inputs
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i].data != nullptr && inputs[i].bytes > 0) {
        int ret = neuron_lib_->SetInput(runtime_, i, inputs[i].data, inputs[i].bytes, buf_attr);
        if (ret != 0) {
          SHERPA_ONNX_LOGE("Failed to set input %zu: %d", i, ret);
          return false;
        }
      }
    }

    // Set outputs
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (outputs[i].data != nullptr && outputs[i].bytes > 0) {
        int ret = neuron_lib_->SetOutput(runtime_, i,
                                         const_cast<void*>(outputs[i].data),
                                         outputs[i].bytes, buf_attr);
        if (ret != 0) {
          SHERPA_ONNX_LOGE("Failed to set output %zu: %d", i, ret);
          return false;
        }
      }
    }

    // Run inference
    int ret = neuron_lib_->Inference(runtime_);
    if (ret != 0) {
      SHERPA_ONNX_LOGE("Neuron inference failed: %d", ret);
      return false;
    }

    return true;
  }

  size_t GetInputTensorSize(size_t index) const {
    if (index >= input_sizes_.size()) {
      return 0;
    }
    return input_sizes_[index];
  }

  size_t GetOutputTensorSize(size_t index) const {
    if (index >= output_sizes_.size()) {
      return 0;
    }
    return output_sizes_[index];
  }

 private:
  void AllocateBuffers() {
    // Calculate input sizes from shapes
    input_sizes_.resize(input_shapes_.size());
    for (size_t i = 0; i < input_shapes_.size(); ++i) {
      size_t size = 4;  // sizeof(float)
      for (auto dim : input_shapes_[i]) {
        size *= dim;
      }
      input_sizes_[i] = size;
    }

    // Calculate output sizes from shapes
    output_sizes_.resize(output_shapes_.size());
    for (size_t i = 0; i < output_shapes_.size(); ++i) {
      size_t size = 4;  // sizeof(float)
      for (auto dim : output_shapes_[i]) {
        size *= dim;
      }
      output_sizes_[i] = size;
    }
  }

  std::string model_path_;
  std::vector<std::vector<uint32_t>> input_shapes_;
  std::vector<std::vector<uint32_t>> output_shapes_;
  std::vector<size_t> input_sizes_;
  std::vector<size_t> output_sizes_;

  std::unique_ptr<mtk::neuropilot::NeuronRuntimeLibrary> neuron_lib_;
  void* runtime_ = nullptr;
};

MtkNpuExecutor::MtkNpuExecutor(const std::string& name)
    : impl_(std::make_unique<Impl>()), name_(name) {}

MtkNpuExecutor::~MtkNpuExecutor() = default;

bool MtkNpuExecutor::Initialize(const std::string& model_path,
                                const std::vector<std::vector<uint32_t>>& input_shapes,
                                const std::vector<std::vector<uint32_t>>& output_shapes,
                                int input_type,
                                int output_type,
                                const std::string& options) {
  is_initialized_ = impl_->Initialize(model_path, input_shapes, output_shapes,
                                       input_type, output_type, options);
  return is_initialized_;
}

bool MtkNpuExecutor::Initialize(const void* model_buffer, size_t model_size,
                                const std::vector<std::vector<uint32_t>>& input_shapes,
                                const std::vector<std::vector<uint32_t>>& output_shapes,
                                int input_type,
                                int output_type,
                                const std::string& options) {
  is_initialized_ = impl_->Initialize(model_buffer, model_size, input_shapes, output_shapes,
                                       input_type, output_type, options);
  return is_initialized_;
}

bool MtkNpuExecutor::RunForMultipleInputsOutputs(
    const std::vector<MtkTensorBuffer>& inputs,
    const std::vector<MtkTensorBuffer>& outputs) {
  return impl_->RunForMultipleInputsOutputs(inputs, outputs);
}

size_t MtkNpuExecutor::GetInputTensorSize(size_t index) const {
  return impl_->GetInputTensorSize(index);
}

size_t MtkNpuExecutor::GetOutputTensorSize(size_t index) const {
  return impl_->GetOutputTensorSize(index);
}

}  // namespace sherpa_onnx
