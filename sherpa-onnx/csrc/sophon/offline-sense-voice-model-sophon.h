// sherpa-onnx/csrc/sophon/offline-sense-voice-model-sophon.h
//
// Copyright (c)  2024  Xiaomi Corporation
// Copyright (c)  2026  Sophon BM1684X backend

#ifndef SHERPA_ONNX_CSRC_SOPHON_OFFLINE_SENSE_VOICE_MODEL_SOPHON_H_
#define SHERPA_ONNX_CSRC_SOPHON_OFFLINE_SENSE_VOICE_MODEL_SOPHON_H_

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

// SenseVoice acoustic model running on a Sophon BM1684X TPU via libbmrt.
// Mirrors OfflineSenseVoiceModelMtk: same Run()/GetModelMetadata() interface so
// OfflineRecognizerSenseVoiceSophonImpl can reuse the SenseVoice frontend,
// CTC greedy decoder and tokenizer unchanged.
class OfflineSenseVoiceModelSophon {
 public:
  ~OfflineSenseVoiceModelSophon();

  explicit OfflineSenseVoiceModelSophon(const OfflineModelConfig& config);

#if __ANDROID_API__ >= 9
  OfflineSenseVoiceModelSophon(AAssetManager* mgr,
                               const OfflineModelConfig& config);
#endif

#if __OHOS__
  OfflineSenseVoiceModelSophon(NativeResourceManager* mgr,
                               const OfflineModelConfig& config);
#endif

  // Run inference
  // @param features Audio features after fbank extraction, shape: [num_frames, 80]
  //                 (before LFR; LFR + pad/truncate happen inside)
  // @param language Language ID (0=auto, 3=zh, 4=en, 7=yue, 11=ja, 12=ko)
  //                 Currently informational: the exported bmodel has the prompt
  //                 baked in and takes a single feature input.
  // @param text_norm Text normalization ID (14=with_itn, 15=without_itn)
  // @return CTC logits, shape: [output_frames, vocab_size]
  std::vector<float> Run(std::vector<float> features, int32_t language,
                         int32_t text_norm) const;

  const OfflineSenseVoiceModelMetaData& GetModelMetadata() const;

  // SenseVoice model constants (defaults; actual shapes are read from the
  // bmodel at load time so a re-export with a different length still works).
  static constexpr int32_t kModelInputFrames = 166;  // ~10 seconds of audio
  static constexpr int32_t kInputFeatDim = 560;      // 80 * 7 (after LFR)
  static constexpr int32_t kVocabSize = 25055;
  static constexpr int32_t kOutputFrames = 170;      // 166 + 4 prompt tokens

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SOPHON_OFFLINE_SENSE_VOICE_MODEL_SOPHON_H_
