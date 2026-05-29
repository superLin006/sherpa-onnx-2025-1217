// sherpa-onnx/csrc/offline-tts-chattts-model-config.cc
//
// Copyright (c)  2026  Sophon BM1684X backend

#include "sherpa-onnx/csrc/offline-tts-chattts-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineTtsChatTtsModelConfig::Register(ParseOptions *po) {
  po->Register("chattts-gpt", &gpt, "Path to the ChatTTS GPT bmodel");
  po->Register("chattts-decoder", &decoder,
               "Path to the ChatTTS DVAE decoder bmodel");
  po->Register("chattts-vocos", &vocos, "Path to the ChatTTS Vocos bmodel");
  po->Register("chattts-vocab", &vocab, "Path to vocab.txt for ChatTTS");
  po->Register("chattts-homophones-map", &homophones_map,
               "Path to homophones_map.json for ChatTTS (optional)");
  po->Register("chattts-speaker-embedding", &speaker_embedding,
               "Path to a speaker embedding binary (768 float16) for zero-shot "
               "timbre. If empty, the built-in default speaker is used.");
  po->Register("chattts-tpu-id", &tpu_id, "Sophon TPU device index");
}

bool OfflineTtsChatTtsModelConfig::Validate() const {
  if (gpt.empty()) {
    SHERPA_ONNX_LOGE("Please provide --chattts-gpt");
    return false;
  }
  if (!FileExists(gpt)) {
    SHERPA_ONNX_LOGE("--chattts-gpt: '%s' does not exist", gpt.c_str());
    return false;
  }

  if (decoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --chattts-decoder");
    return false;
  }
  if (!FileExists(decoder)) {
    SHERPA_ONNX_LOGE("--chattts-decoder: '%s' does not exist", decoder.c_str());
    return false;
  }

  if (vocos.empty()) {
    SHERPA_ONNX_LOGE("Please provide --chattts-vocos");
    return false;
  }
  if (!FileExists(vocos)) {
    SHERPA_ONNX_LOGE("--chattts-vocos: '%s' does not exist", vocos.c_str());
    return false;
  }

  if (vocab.empty()) {
    SHERPA_ONNX_LOGE("Please provide --chattts-vocab");
    return false;
  }
  if (!FileExists(vocab)) {
    SHERPA_ONNX_LOGE("--chattts-vocab: '%s' does not exist", vocab.c_str());
    return false;
  }

  if (!homophones_map.empty() && !FileExists(homophones_map)) {
    SHERPA_ONNX_LOGE("--chattts-homophones-map: '%s' does not exist",
                     homophones_map.c_str());
    return false;
  }

  if (!speaker_embedding.empty() && !FileExists(speaker_embedding)) {
    SHERPA_ONNX_LOGE("--chattts-speaker-embedding: '%s' does not exist",
                     speaker_embedding.c_str());
    return false;
  }

  return true;
}

std::string OfflineTtsChatTtsModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineTtsChatTtsModelConfig(";
  os << "gpt=\"" << gpt << "\", ";
  os << "decoder=\"" << decoder << "\", ";
  os << "vocos=\"" << vocos << "\", ";
  os << "vocab=\"" << vocab << "\", ";
  os << "homophones_map=\"" << homophones_map << "\", ";
  os << "speaker_embedding=\"" << speaker_embedding << "\", ";
  os << "tpu_id=" << tpu_id << ")";

  return os.str();
}

}  // namespace sherpa_onnx
