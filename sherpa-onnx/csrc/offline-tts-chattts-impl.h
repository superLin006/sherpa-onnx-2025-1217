// sherpa-onnx/csrc/offline-tts-chattts-impl.h
//
// Copyright (c)  2026  Sophon BM1684X backend

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_CHATTTS_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_CHATTTS_IMPL_H_

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-tts-impl.h"
// ChatTTS engine (vendored under csrc/sophon/chattts/)
#include "sherpa-onnx/csrc/sophon/chattts/chattts.h"

namespace sherpa_onnx {

class OfflineTtsChatTtsImpl : public OfflineTtsImpl {
 public:
  explicit OfflineTtsChatTtsImpl(const OfflineTtsConfig &config)
      : config_(config) {
    const auto &c = config.model.chattts;

    ChatTTSConfig engine_cfg;
    engine_cfg.gpt_model_path = c.gpt;
    engine_cfg.decoder_model_path = c.decoder;
    engine_cfg.vocos_model_path = c.vocos;
    engine_cfg.vocab_path = c.vocab;
    engine_cfg.homophones_map_path = c.homophones_map;
    engine_cfg.tpu_id = c.tpu_id;
    engine_cfg.sample_rate = sample_rate_;

    engine_ = std::make_unique<ChatTTS>(engine_cfg);

    if (!c.speaker_embedding.empty()) {
      if (!engine_->load_speaker(c.speaker_embedding)) {
        SHERPA_ONNX_LOGE(
            "ChatTTS: failed to load speaker embedding '%s'; using the "
            "built-in default speaker.",
            c.speaker_embedding.c_str());
      }
    }
  }

  template <typename Manager>
  OfflineTtsChatTtsImpl(Manager * /*mgr*/, const OfflineTtsConfig &config)
      : OfflineTtsChatTtsImpl(config) {}

  int32_t SampleRate() const override { return sample_rate_; }

  int32_t NumSpeakers() const override { return 1; }

  GeneratedAudio Generate(const std::string &text, int64_t /*sid*/ = 0,
                          float speed = 1.0,
                          GeneratedAudioCallback callback = nullptr) const override {
    InferParams params;
    // sherpa speed is a multiplier (1.0 = normal). ChatTTS uses a [speed_N]
    // tag in [1, 9] with 5 == normal. Map and clamp.
    int32_t tag = static_cast<int32_t>(std::lround(5.0f * speed));
    params.speed = std::max(1, std::min(9, tag));

    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<float> samples;
    if (callback) {
      // Stream chunks to the caller; also accumulate for the return value.
      StreamParams sparams;
      float total_hint = 0.0f;  // ChatTTS does not expose a progress ratio
      engine_->infer_stream(
          text, params, sparams,
          [&samples, &callback, &total_hint](const std::vector<float> &chunk) {
            samples.insert(samples.end(), chunk.begin(), chunk.end());
            // progress is unknown ahead of time; report a monotonically
            // increasing placeholder based on produced samples.
            total_hint += 0.0f;
            callback(chunk.data(), static_cast<int32_t>(chunk.size()), 1.0f);
          },
          /*do_normalize=*/true);
    } else {
      samples = engine_->infer(text, params, /*do_normalize=*/true);
    }

    GeneratedAudio ans;
    ans.samples = std::move(samples);
    ans.sample_rate = sample_rate_;
    return ans;
  }

 private:
  OfflineTtsConfig config_;
  int32_t sample_rate_ = 24000;  // ChatTTS fixed output rate

  // ChatTTS::infer / infer_stream are non-const (mutate engine state) while
  // Generate() is const; hold the engine mutably and serialize calls.
  mutable std::unique_ptr<ChatTTS> engine_;
  mutable std::mutex mutex_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_CHATTTS_IMPL_H_
