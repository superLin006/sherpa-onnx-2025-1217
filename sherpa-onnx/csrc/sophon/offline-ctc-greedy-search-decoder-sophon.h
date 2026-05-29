// sherpa-onnx/csrc/sophon/offline-ctc-greedy-search-decoder-sophon.h
//
// Copyright (c)  2024  Xiaomi Corporation
// Copyright (c)  2026  Sophon BM1684X backend

#ifndef SHERPA_ONNX_CSRC_SOPHON_OFFLINE_CTC_GREEDY_SEARCH_DECODER_SOPHON_H_
#define SHERPA_ONNX_CSRC_SOPHON_OFFLINE_CTC_GREEDY_SEARCH_DECODER_SOPHON_H_

#include <cstdint>

#include "sherpa-onnx/csrc/offline-ctc-decoder.h"

namespace sherpa_onnx {

// Plain CTC greedy search over flat logits [num_frames, vocab_size].
// Self-contained so the Sophon backend does not depend on other backends.
class OfflineCtcGreedySearchDecoderSophon {
 public:
  explicit OfflineCtcGreedySearchDecoderSophon(int32_t blank_id)
      : blank_id_(blank_id) {}

  OfflineCtcDecoderResult Decode(const float *logits, int32_t num_frames,
                                 int32_t vocab_size) {
    OfflineCtcDecoderResult result;
    result.tokens.reserve(num_frames);
    result.timestamps.reserve(num_frames);

    int32_t prev_token = -1;
    for (int32_t t = 0; t < num_frames; ++t) {
      int32_t max_token = 0;
      float max_logp = logits[t * vocab_size];

      for (int32_t i = 1; i < vocab_size; ++i) {
        if (logits[t * vocab_size + i] > max_logp) {
          max_logp = logits[t * vocab_size + i];
          max_token = i;
        }
      }

      // CTC decoding rule: skip blank and repeated tokens
      if (max_token != blank_id_ && max_token != prev_token) {
        result.tokens.push_back(max_token);
        result.timestamps.push_back(t);
      }
      prev_token = max_token;
    }

    return result;
  }

 private:
  int32_t blank_id_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SOPHON_OFFLINE_CTC_GREEDY_SEARCH_DECODER_SOPHON_H_
