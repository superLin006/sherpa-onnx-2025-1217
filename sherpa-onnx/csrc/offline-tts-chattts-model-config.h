// sherpa-onnx/csrc/offline-tts-chattts-model-config.h
//
// Copyright (c)  2026  Sophon BM1684X backend

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_CHATTTS_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_CHATTTS_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

// ChatTTS running on a Sophon BM1684X TPU (GPT + DVAE decoder + Vocos + iSTFT).
struct OfflineTtsChatTtsModelConfig {
  // bmodel paths
  std::string gpt;       // GPT autoregressive model
  std::string decoder;   // DVAE decoder
  std::string vocos;     // Vocos vocoder

  // assets
  std::string vocab;            // vocab.txt
  std::string homophones_map;   // homophones_map.json (optional)

  // Zero-shot speaker timbre: a binary file of speaker embedding
  // (768 float16). If empty, the engine uses its built-in default speaker.
  std::string speaker_embedding;

  // Sophon TPU device index
  int32_t tpu_id = 0;

  OfflineTtsChatTtsModelConfig() = default;

  OfflineTtsChatTtsModelConfig(const std::string &gpt,
                               const std::string &decoder,
                               const std::string &vocos,
                               const std::string &vocab,
                               const std::string &homophones_map,
                               const std::string &speaker_embedding,
                               int32_t tpu_id)
      : gpt(gpt),
        decoder(decoder),
        vocos(vocos),
        vocab(vocab),
        homophones_map(homophones_map),
        speaker_embedding(speaker_embedding),
        tpu_id(tpu_id) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_CHATTTS_MODEL_CONFIG_H_
