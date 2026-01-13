// sherpa-onnx/csrc/mtk/offline-sense-voice-model-mtk.cc
//
// Copyright (c)  2024  Xiaomi Corporation
// Copyright (c)  2024  MediaTek Inc.

#include "sherpa-onnx/csrc/mtk/offline-sense-voice-model-mtk.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/log.h"
#include "sherpa-onnx/csrc/mtk/macros.h"
#include "sherpa-onnx/csrc/mtk/mtk-npu-executor.h"

// Neuron tensor type constants (from NeuronAdapter.h)
// We define them locally to avoid dependency on NeuronAdapter.h
// since we use dynamic loading
constexpr int NEURON_TENSOR_FLOAT32 = 3;

namespace sherpa_onnx {

class OfflineSenseVoiceModelMtk::Impl {
 public:
  explicit Impl(const OfflineModelConfig& config) : config_(config) {
    Init(config);
  }

#if __ANDROID_API__ >= 9
  Impl(AAssetManager* mgr, const OfflineModelConfig& config)
      : config_(config), asset_manager_(mgr) {
    Init(config);
  }
#endif

#if __OHOS__
  Impl(NativeResourceManager* mgr, const OfflineModelConfig& config)
      : config_(config), resource_manager_(mgr) {
    Init(config);
  }
#endif

  std::vector<float> Run(std::vector<float> features,
                         int32_t language,
                         int32_t text_norm) const {
    // Apply LFR (Low Frame Rate) transformation
    // Input: [num_frames, 80] -> Output: [lfr_frames, 560]
    features = ApplyLFR(std::move(features));

    int32_t num_lfr_frames = features.size() / kInputFeatDim;
    SHERPA_ONNX_LOGI("LFR frames: %d", num_lfr_frames);

    // Prepare padded features for fixed model input
    std::vector<float> padded_features(kModelInputFrames * kInputFeatDim, 0.0f);
    int32_t frames_to_copy = std::min(num_lfr_frames, kModelInputFrames);
    std::memcpy(padded_features.data(), features.data(),
                frames_to_copy * kInputFeatDim * sizeof(float));

    // Prepare prompt tensors (language_id, event_id, event_type_id, text_norm_id)
    std::vector<float> language_tensor = {static_cast<float>(language)};
    std::vector<float> event_tensor = {1.0f};       // Fixed event_id
    std::vector<float> event_type_tensor = {2.0f};  // Fixed event_type_id
    std::vector<float> text_norm_tensor = {static_cast<float>(text_norm)};

    // Prepare output buffer
    std::vector<float> output(kOutputFrames * kVocabSize, 0.0f);

    // Create input/output buffers
    std::vector<MtkTensorBuffer> inputs(5);
    inputs[0] = {padded_features.data(),
                 padded_features.size() * sizeof(float),
                 MtkTensorDataType::kFloat32};
    inputs[1] = {language_tensor.data(),
                 language_tensor.size() * sizeof(float),
                 MtkTensorDataType::kFloat32};
    inputs[2] = {event_tensor.data(),
                 event_tensor.size() * sizeof(float),
                 MtkTensorDataType::kFloat32};
    inputs[3] = {event_type_tensor.data(),
                 event_type_tensor.size() * sizeof(float),
                 MtkTensorDataType::kFloat32};
    inputs[4] = {text_norm_tensor.data(),
                 text_norm_tensor.size() * sizeof(float),
                 MtkTensorDataType::kFloat32};

    std::vector<MtkTensorBuffer> outputs(1);
    outputs[0] = {output.data(),
                  output.size() * sizeof(float),
                  MtkTensorDataType::kFloat32};

    // Run inference
    bool success = executor_->RunForMultipleInputsOutputs(inputs, outputs);
    if (!success) {
      SHERPA_ONNX_LOGE("MTK NPU inference failed");
      return {};
    }

    // Return only valid frames (frames_to_copy + 4 prompt tokens)
    int32_t valid_output_frames = frames_to_copy + 4;
    std::vector<float> result(valid_output_frames * kVocabSize);
    std::memcpy(result.data(), output.data(),
                valid_output_frames * kVocabSize * sizeof(float));

    return result;
  }

  const OfflineSenseVoiceModelMetaData& GetModelMetadata() const {
    return meta_data_;
  }

 private:
  void Init(const OfflineModelConfig& config) {
    // Initialize metadata
    InitMetaData();

    // Get model path
    std::string model_path = config.sense_voice.model;
    SHERPA_ONNX_LOGI("Loading MTK SenseVoice model: %s", model_path.c_str());

    // Create executor
    executor_ = std::make_unique<MtkNpuExecutor>("sensevoice_mtk");

    // Define input/output shapes for SenseVoice model
    // Input 0: Audio features [1, 166, 560]
    // Input 1-4: prompt IDs [1]
    // Output: CTC logits [1, 170, 25055]
    std::vector<std::vector<uint32_t>> input_shapes = {
        {1, kModelInputFrames, kInputFeatDim},  // Audio features
        {1},  // language_id
        {1},  // event_id
        {1},  // event_type_id
        {1}   // text_norm_id
    };

    std::vector<std::vector<uint32_t>> output_shapes = {
        {1, kOutputFrames, kVocabSize}  // CTC logits
    };

    bool success = executor_->Initialize(
        model_path,
        input_shapes,
        output_shapes,
        NEURON_TENSOR_FLOAT32,
        NEURON_TENSOR_FLOAT32);

    if (!success) {
      SHERPA_ONNX_LOGE("Failed to initialize MTK NPU executor");
      return;
    }

    SHERPA_ONNX_LOGI("MTK SenseVoice model loaded successfully");
  }

  void InitMetaData() {
    // Initialize SenseVoice model metadata
    meta_data_.with_itn_id = 14;
    meta_data_.without_itn_id = 15;
    meta_data_.window_size = 7;
    meta_data_.window_shift = 6;
    meta_data_.vocab_size = kVocabSize;
    meta_data_.normalize_samples = 0;

    // Language ID mapping
    meta_data_.lang2id = {
        {"auto", 0},
        {"zh", 3},
        {"en", 4},
        {"yue", 7},
        {"ja", 11},
        {"ko", 12}
    };
  }

  std::vector<float> ApplyLFR(std::vector<float> in) const {
    // LFR: Low Frame Rate transformation
    // Stack 7 consecutive frames with shift of 6
    // Input: [num_frames, 80] -> Output: [lfr_frames, 560]

    const int32_t feat_dim = 80;
    int32_t num_frames = in.size() / feat_dim;

    if (num_frames < meta_data_.window_size) {
      SHERPA_ONNX_LOGW("Input frames (%d) less than window size (%d)",
                       num_frames, meta_data_.window_size);
      // Pad with zeros if necessary
      int32_t pad_frames = meta_data_.window_size - num_frames;
      in.resize(meta_data_.window_size * feat_dim, 0.0f);
      num_frames = meta_data_.window_size;
    }

    int32_t out_num_frames = (num_frames - meta_data_.window_size) /
                                 meta_data_.window_shift + 1;

    std::vector<float> out(out_num_frames * kInputFeatDim);

    for (int32_t i = 0; i < out_num_frames; ++i) {
      int32_t start_frame = i * meta_data_.window_shift;
      for (int32_t j = 0; j < meta_data_.window_size; ++j) {
        std::memcpy(out.data() + i * kInputFeatDim + j * feat_dim,
                    in.data() + (start_frame + j) * feat_dim,
                    feat_dim * sizeof(float));
      }
    }

    return out;
  }

  OfflineModelConfig config_;
  OfflineSenseVoiceModelMetaData meta_data_;
  std::unique_ptr<MtkNpuExecutor> executor_;

#if __ANDROID_API__ >= 9
  AAssetManager* asset_manager_ = nullptr;
#endif

#if __OHOS__
  NativeResourceManager* resource_manager_ = nullptr;
#endif
};

OfflineSenseVoiceModelMtk::~OfflineSenseVoiceModelMtk() = default;

OfflineSenseVoiceModelMtk::OfflineSenseVoiceModelMtk(const OfflineModelConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

#if __ANDROID_API__ >= 9
OfflineSenseVoiceModelMtk::OfflineSenseVoiceModelMtk(AAssetManager* mgr,
                                                     const OfflineModelConfig& config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}
#endif

#if __OHOS__
OfflineSenseVoiceModelMtk::OfflineSenseVoiceModelMtk(NativeResourceManager* mgr,
                                                     const OfflineModelConfig& config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}
#endif

std::vector<float> OfflineSenseVoiceModelMtk::Run(std::vector<float> features,
                                                   int32_t language,
                                                   int32_t text_norm) const {
  return impl_->Run(std::move(features), language, text_norm);
}

const OfflineSenseVoiceModelMetaData& OfflineSenseVoiceModelMtk::GetModelMetadata() const {
  return impl_->GetModelMetadata();
}

}  // namespace sherpa_onnx
