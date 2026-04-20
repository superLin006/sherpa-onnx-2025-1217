// sherpa-onnx/csrc/mtk/online-zipformer-transducer-model-mtk.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_MTK_ONLINE_ZIPFORMER_TRANSDUCER_MODEL_MTK_H_
#define SHERPA_ONNX_CSRC_MTK_ONLINE_ZIPFORMER_TRANSDUCER_MODEL_MTK_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/online-model-config.h"

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

namespace sherpa_onnx {

// Encoder cached state storage for streaming zipformer on MTK NPU.
// All states are stored as float32 vectors.
//
// The zipformer encoder has 5 stacks with downsampling factors (1,2,4,8,2).
// Each stack has 2 encoder layers.
// For each stack, we cache: cached_len, cached_avg, cached_key, cached_val,
// cached_val2, cached_conv1, cached_conv2 — total 35 states.
struct EncoderCacheMtk {
  static constexpr int32_t kNumCachedStates = 35;

  // Sizes in number of float elements for each cached state.
  // Order: cached_len(5), cached_avg(5), cached_key(5), cached_val(5),
  //        cached_val2(5), cached_conv1(5), cached_conv2(5)
  static constexpr int32_t kCacheStateSizes[kNumCachedStates] = {
      // cached_len (5): [2,1] = 2 each
      2, 2, 2, 2, 2,
      // cached_avg (5): [2,1,256] = 512 each
      512, 512, 512, 512, 512,
      // cached_key (5): left_ctx//ds * attention_dim * num_layers
      49152, 24576, 12288, 6144, 24576,
      // cached_val (5): left_ctx//ds * (attention_dim//2) * num_layers
      24576, 12288, 6144, 3072, 12288,
      // cached_val2 (5): same as cached_val
      24576, 12288, 6144, 3072, 12288,
      // cached_conv1 (5): [2,1,256,30] = 15360 each
      15360, 15360, 15360, 15360, 15360,
      // cached_conv2 (5): [2,1,256,30] = 15360 each
      15360, 15360, 15360, 15360, 15360,
  };

  std::vector<std::vector<float>> buf;

  EncoderCacheMtk() { Reset(); }

  void Reset() {
    buf.resize(kNumCachedStates);
    for (int32_t i = 0; i < kNumCachedStates; ++i) {
      buf[i].assign(kCacheStateSizes[i], 0.0f);
    }
  }

  float *Data(int32_t i) { return buf[i].data(); }
  const float *Data(int32_t i) const { return buf[i].data(); }
  size_t Bytes(int32_t i) const { return buf[i].size() * sizeof(float); }
};

class OnlineZipformerTransducerModelMtk {
 public:
  ~OnlineZipformerTransducerModelMtk();

  explicit OnlineZipformerTransducerModelMtk(const OnlineModelConfig &config);

#if __ANDROID_API__ >= 9
  OnlineZipformerTransducerModelMtk(AAssetManager *mgr,
                                    const OnlineModelConfig &config);
#endif

#if __OHOS__
  OnlineZipformerTransducerModelMtk(NativeResourceManager *mgr,
                                    const OnlineModelConfig &config);
#endif

  // Get initial encoder states (all zeros)
  std::vector<std::vector<float>> GetEncoderInitStates() const;

  // Run encoder: features + cached states -> encoder_out + new cached states
  // features: [1, chunk_size, 80] flattened
  // Returns: (encoder_out flattened [1, encoder_out_frames, encoder_dim],
  //           new cached states)
  std::pair<std::vector<float>, std::vector<std::vector<float>>> RunEncoder(
      const std::vector<float> &features,
      std::vector<std::vector<float>> states);

  // Run decoder: token IDs -> decoder_out
  // Internally performs CPU embedding lookup, then runs decoder NPU model.
  // tokens: last context_size token IDs
  std::vector<float> RunDecoder(const std::vector<int64_t> &tokens);

  // Run joiner: encoder_frame + decoder_out -> logits
  // encoder_out: [encoder_dim] float pointer
  // decoder_out: [decoder_dim] float pointer
  // Returns: logits [vocab_size]
  std::vector<float> RunJoiner(const float *encoder_out,
                               const float *decoder_out);

  int32_t ContextSize() const;
  int32_t ChunkSize() const;       // SEGMENT = 103
  int32_t ChunkShift() const;      // OFFSET = 96
  int32_t VocabSize() const;
  int32_t EncoderOutDim() const;    // ENC_DIM = 256
  int32_t EncoderOutFrames() const; // ENCODER_OUT_T = 24

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_MTK_ONLINE_ZIPFORMER_TRANSDUCER_MODEL_MTK_H_
