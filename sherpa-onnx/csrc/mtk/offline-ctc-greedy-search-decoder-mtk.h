// sherpa-onnx/csrc/mtk/offline-ctc-greedy-search-decoder-mtk.h
//
// Copyright (c)  2024  Xiaomi Corporation
// Copyright (c)  2024  MediaTek Inc.

#ifndef SHERPA_ONNX_CSRC_MTK_OFFLINE_CTC_GREEDY_SEARCH_DECODER_MTK_H_
#define SHERPA_ONNX_CSRC_MTK_OFFLINE_CTC_GREEDY_SEARCH_DECODER_MTK_H_

#include <vector>
#include "sherpa-onnx/csrc/offline-ctc-decoder.h"

namespace sherpa_onnx {

class OfflineCtcGreedySearchDecoderMtk {
 public:
  explicit OfflineCtcGreedySearchDecoderMtk(int32_t blank_id)
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

#endif  // SHERPA_ONNX_CSRC_MTK_OFFLINE_CTC_GREEDY_SEARCH_DECODER_MTK_H_
