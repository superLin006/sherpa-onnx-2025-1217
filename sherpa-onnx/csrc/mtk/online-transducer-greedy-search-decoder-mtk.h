// sherpa-onnx/csrc/mtk/online-transducer-greedy-search-decoder-mtk.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_MTK_ONLINE_TRANSDUCER_GREEDY_SEARCH_DECODER_MTK_H_
#define SHERPA_ONNX_CSRC_MTK_ONLINE_TRANSDUCER_GREEDY_SEARCH_DECODER_MTK_H_

#include <vector>

#include "sherpa-onnx/csrc/mtk/online-transducer-decoder-mtk.h"
#include "sherpa-onnx/csrc/mtk/online-zipformer-transducer-model-mtk.h"

namespace sherpa_onnx {

class OnlineTransducerGreedySearchDecoderMtk
    : public OnlineTransducerDecoderMtk {
 public:
  explicit OnlineTransducerGreedySearchDecoderMtk(
      OnlineZipformerTransducerModelMtk *model, int32_t unk_id = 2,
      float blank_penalty = 0.0)
      : model_(model), unk_id_(unk_id), blank_penalty_(blank_penalty) {}

  OnlineTransducerDecoderResultMtk GetEmptyResult() const override;

  void StripLeadingBlanks(OnlineTransducerDecoderResultMtk *r) const override;

  void Decode(std::vector<float> encoder_out,
              OnlineTransducerDecoderResultMtk *result) const override;

 private:
  OnlineZipformerTransducerModelMtk *model_;  // Not owned
  int32_t unk_id_;
  float blank_penalty_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_MTK_ONLINE_TRANSDUCER_GREEDY_SEARCH_DECODER_MTK_H_
