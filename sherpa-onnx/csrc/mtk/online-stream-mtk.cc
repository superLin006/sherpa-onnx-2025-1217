// sherpa-onnx/csrc/mtk/online-stream-mtk.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/mtk/online-stream-mtk.h"

#include <utility>
#include <vector>

namespace sherpa_onnx {

class OnlineStreamMtk::Impl {
 public:
  void SetEncoderStates(std::vector<std::vector<float>> states) {
    states_ = std::move(states);
  }

  std::vector<std::vector<float>> &GetEncoderStates() { return states_; }

  void SetResult(OnlineTransducerDecoderResultMtk r) {
    result_ = std::move(r);
  }

  OnlineTransducerDecoderResultMtk &GetResult() { return result_; }

 private:
  std::vector<std::vector<float>> states_;
  OnlineTransducerDecoderResultMtk result_;
};

OnlineStreamMtk::OnlineStreamMtk(const FeatureExtractorConfig &config /*= {}*/,
                                 ContextGraphPtr context_graph /*= nullptr*/)
    : OnlineStream(config, context_graph), impl_(std::make_unique<Impl>()) {}

OnlineStreamMtk::~OnlineStreamMtk() = default;

void OnlineStreamMtk::SetEncoderStates(
    std::vector<std::vector<float>> states) const {
  impl_->SetEncoderStates(std::move(states));
}

std::vector<std::vector<float>> &OnlineStreamMtk::GetEncoderStates() const {
  return impl_->GetEncoderStates();
}

void OnlineStreamMtk::SetResult(OnlineTransducerDecoderResultMtk r) const {
  impl_->SetResult(std::move(r));
}

OnlineTransducerDecoderResultMtk &OnlineStreamMtk::GetResult() const {
  return impl_->GetResult();
}

}  // namespace sherpa_onnx
