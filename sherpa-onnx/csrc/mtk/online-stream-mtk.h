// sherpa-onnx/csrc/mtk/online-stream-mtk.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_MTK_ONLINE_STREAM_MTK_H_
#define SHERPA_ONNX_CSRC_MTK_ONLINE_STREAM_MTK_H_

#include <memory>
#include <vector>

#include "sherpa-onnx/csrc/online-stream.h"
#include "sherpa-onnx/csrc/mtk/online-transducer-decoder-mtk.h"

namespace sherpa_onnx {

class OnlineStreamMtk : public OnlineStream {
 public:
  explicit OnlineStreamMtk(const FeatureExtractorConfig &config = {},
                           ContextGraphPtr context_graph = nullptr);

  ~OnlineStreamMtk();

  void SetEncoderStates(std::vector<std::vector<float>> states) const;

  std::vector<std::vector<float>> &GetEncoderStates() const;

  void SetResult(OnlineTransducerDecoderResultMtk r) const;

  OnlineTransducerDecoderResultMtk &GetResult() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_MTK_ONLINE_STREAM_MTK_H_
