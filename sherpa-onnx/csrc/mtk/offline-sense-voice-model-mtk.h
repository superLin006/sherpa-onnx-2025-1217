// sherpa-onnx/csrc/mtk/offline-sense-voice-model-mtk.h
//
// Copyright (c)  2024  Xiaomi Corporation
// Copyright (c)  2024  MediaTek Inc.

#ifndef SHERPA_ONNX_CSRC_MTK_OFFLINE_SENSE_VOICE_MODEL_MTK_H_
#define SHERPA_ONNX_CSRC_MTK_OFFLINE_SENSE_VOICE_MODEL_MTK_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/csrc/offline-sense-voice-model-meta-data.h"

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

namespace sherpa_onnx {

class OfflineSenseVoiceModelMtk {
 public:
  ~OfflineSenseVoiceModelMtk();

  explicit OfflineSenseVoiceModelMtk(const OfflineModelConfig& config);

#if __ANDROID_API__ >= 9
  OfflineSenseVoiceModelMtk(AAssetManager* mgr, const OfflineModelConfig& config);
#endif

#if __OHOS__
  OfflineSenseVoiceModelMtk(NativeResourceManager* mgr, const OfflineModelConfig& config);
#endif

  // Run inference
  // @param features Audio features after fbank extraction, shape: [num_frames, 80]
  // @param language Language ID (0=auto, 3=zh, 4=en, 7=yue, 11=ja, 12=ko)
  // @param text_norm Text normalization ID (14=with_itn, 15=without_itn)
  // @return CTC logits, shape: [valid_frames, vocab_size]
  std::vector<float> Run(std::vector<float> features,
                         int32_t language,
                         int32_t text_norm) const;

  const OfflineSenseVoiceModelMetaData& GetModelMetadata() const;

  // SenseVoice model constants
  static constexpr int32_t kModelInputFrames = 166;   // ~10 seconds of audio
  static constexpr int32_t kInputFeatDim = 560;       // 80 * 7 (after LFR)
  static constexpr int32_t kVocabSize = 25055;
  static constexpr int32_t kOutputFrames = 170;       // 166 + 4 prompt tokens

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_MTK_OFFLINE_SENSE_VOICE_MODEL_MTK_H_
