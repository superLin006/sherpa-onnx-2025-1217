// sherpa-onnx/csrc/mtk/online-transducer-modified-beam-search-decoder-mtk.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_MTK_ONLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_DECODER_MTK_H_
#define SHERPA_ONNX_CSRC_MTK_ONLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_DECODER_MTK_H_

#include <vector>

#include "sherpa-onnx/csrc/context-graph.h"
#include "sherpa-onnx/csrc/mtk/online-transducer-decoder-mtk.h"
#include "sherpa-onnx/csrc/mtk/online-zipformer-transducer-model-mtk.h"
#include "sherpa-onnx/csrc/online-stream.h"

namespace sherpa_onnx {

class OnlineTransducerModifiedBeamSearchDecoderMtk
    : public OnlineTransducerDecoderMtk {
 public:
  explicit OnlineTransducerModifiedBeamSearchDecoderMtk(
      OnlineZipformerTransducerModelMtk *model, int32_t max_active_paths,
      int32_t unk_id = 2, float blank_penalty = 0.0)
      : model_(model),
        max_active_paths_(max_active_paths),
        unk_id_(unk_id),
        blank_penalty_(blank_penalty) {}

  OnlineTransducerDecoderResultMtk GetEmptyResult() const override;

  void StripLeadingBlanks(OnlineTransducerDecoderResultMtk *r) const override;

  void Decode(std::vector<float> encoder_out,
              OnlineTransducerDecoderResultMtk *result) const override;

  // Decode with hotword context graph support
  void Decode(std::vector<float> encoder_out,
              OnlineTransducerDecoderResultMtk *result,
              OnlineStream *stream) const;

 private:
  OnlineZipformerTransducerModelMtk *model_;  // Not owned
  int32_t max_active_paths_;
  int32_t unk_id_;
  float blank_penalty_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_MTK_ONLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_DECODER_MTK_H_
