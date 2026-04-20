// sherpa-onnx/csrc/mtk/online-transducer-greedy-search-decoder-mtk.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/mtk/online-transducer-greedy-search-decoder-mtk.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

OnlineTransducerDecoderResultMtk
OnlineTransducerGreedySearchDecoderMtk::GetEmptyResult() const {
  int32_t context_size = model_->ContextSize();
  int32_t blank_id = 0;  // always 0
  OnlineTransducerDecoderResultMtk r;
  r.tokens.resize(context_size, -1);
  r.tokens.back() = blank_id;

  return r;
}

void OnlineTransducerGreedySearchDecoderMtk::StripLeadingBlanks(
    OnlineTransducerDecoderResultMtk *r) const {
  int32_t context_size = model_->ContextSize();

  auto start = r->tokens.begin() + context_size;
  auto end = r->tokens.end();

  r->tokens = std::vector<int64_t>(start, end);
}

void OnlineTransducerGreedySearchDecoderMtk::Decode(
    std::vector<float> encoder_out,
    OnlineTransducerDecoderResultMtk *result) const {
  auto &r = result[0];
  int32_t num_frames = model_->EncoderOutFrames();
  int32_t encoder_out_dim = model_->EncoderOutDim();
  int32_t vocab_size = model_->VocabSize();
  int32_t context_size = model_->ContextSize();

  std::vector<int64_t> decoder_input;
  std::vector<float> decoder_out;

  if (r.previous_decoder_out.empty()) {
    decoder_input = {r.tokens.begin() + (r.tokens.size() - context_size),
                     r.tokens.end()};
    decoder_out = model_->RunDecoder(decoder_input);
  } else {
    decoder_out = std::move(r.previous_decoder_out);
  }

  const float *p_encoder_out = encoder_out.data();
  for (int32_t t = 0; t != num_frames; ++t) {
    auto logit = model_->RunJoiner(p_encoder_out, decoder_out.data());
    p_encoder_out += encoder_out_dim;

    if (blank_penalty_ > 0.0) {
      logit[0] -= blank_penalty_;  // assuming blank id is 0
    }

    auto y = static_cast<int32_t>(std::distance(
        logit.data(),
        std::max_element(logit.data(), logit.data() + vocab_size)));

    // blank id is hardcoded to 0
    // also, it treats unk as blank
    if (y != 0 && y != unk_id_) {
      r.tokens.push_back(y);
      r.timestamps.push_back(t + r.frame_offset);
      r.num_trailing_blanks = 0;

      decoder_input = {r.tokens.begin() + (r.tokens.size() - context_size),
                       r.tokens.end()};
      decoder_out = model_->RunDecoder(decoder_input);
    } else {
      ++r.num_trailing_blanks;
    }
  }

  r.frame_offset += num_frames;
  r.previous_decoder_out = std::move(decoder_out);
}

}  // namespace sherpa_onnx
