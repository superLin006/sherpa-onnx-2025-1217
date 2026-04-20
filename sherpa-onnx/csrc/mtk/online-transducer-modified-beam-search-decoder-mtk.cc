// sherpa-onnx/csrc/mtk/online-transducer-modified-beam-search-decoder-mtk.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/mtk/online-transducer-modified-beam-search-decoder-mtk.h"

#include <algorithm>
#include <tuple>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/hypothesis.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"

namespace sherpa_onnx {

OnlineTransducerDecoderResultMtk
OnlineTransducerModifiedBeamSearchDecoderMtk::GetEmptyResult() const {
  int32_t context_size = model_->ContextSize();
  int32_t blank_id = 0;  // always 0
  OnlineTransducerDecoderResultMtk r;

  std::vector<int64_t> blanks(context_size, -1);
  blanks.back() = blank_id;

  Hypotheses blank_hyp({{blanks, 0}});
  r.hyps = std::move(blank_hyp);
  r.tokens = std::move(blanks);

  return r;
}

void OnlineTransducerModifiedBeamSearchDecoderMtk::StripLeadingBlanks(
    OnlineTransducerDecoderResultMtk *r) const {
  int32_t context_size = model_->ContextSize();
  auto hyp = r->hyps.GetMostProbable(true);

  std::vector<int64_t> tokens(hyp.ys.begin() + context_size, hyp.ys.end());
  r->tokens = std::move(tokens);
  r->timestamps = std::move(hyp.timestamps);

  r->num_trailing_blanks = hyp.num_trailing_blanks;
}

// Helper: run decoder for each hypothesis (serial, one at a time)
static std::vector<std::vector<float>> GetDecoderOut(
    OnlineZipformerTransducerModelMtk *model, const Hypotheses &hyp_vec) {
  std::vector<std::vector<float>> ans;
  ans.reserve(hyp_vec.Size());

  int32_t context_size = model->ContextSize();
  for (const auto &p : hyp_vec) {
    const auto &hyp = p.second;
    auto start = hyp.ys.begin() + (hyp.ys.size() - context_size);
    auto end = hyp.ys.end();
    std::vector<int64_t> tokens(start, end);
    auto decoder_out = model->RunDecoder(tokens);
    ans.push_back(std::move(decoder_out));
  }

  return ans;
}

// Helper: run joiner for each hypothesis and apply log-softmax
static std::vector<std::vector<float>> GetJoinerOutLogSoftmax(
    OnlineZipformerTransducerModelMtk *model, const float *p_encoder_out,
    const std::vector<std::vector<float>> &decoder_out) {
  std::vector<std::vector<float>> ans;
  ans.reserve(decoder_out.size());

  for (const auto &d : decoder_out) {
    auto joiner_out = model->RunJoiner(p_encoder_out, d.data());
    LogSoftmax(joiner_out.data(), joiner_out.size());
    ans.push_back(std::move(joiner_out));
  }
  return ans;
}

void OnlineTransducerModifiedBeamSearchDecoderMtk::Decode(
    std::vector<float> encoder_out,
    OnlineTransducerDecoderResultMtk *result) const {
  // No context graph — delegate to the version without stream
  Decode(std::move(encoder_out), result, nullptr);
}

void OnlineTransducerModifiedBeamSearchDecoderMtk::Decode(
    std::vector<float> encoder_out, OnlineTransducerDecoderResultMtk *result,
    OnlineStream *stream) const {
  auto &r = result[0];
  int32_t num_frames = model_->EncoderOutFrames();
  int32_t encoder_out_dim = model_->EncoderOutDim();
  int32_t vocab_size = model_->VocabSize();
  int32_t context_size = model_->ContextSize();

  Hypotheses cur = std::move(result->hyps);
  std::vector<Hypothesis> prev;

  auto decoder_out = std::move(result->previous_decoder_out2);
  if (decoder_out.empty()) {
    decoder_out = GetDecoderOut(model_, cur);
  }

  const float *p_encoder_out = encoder_out.data();

  int32_t frame_offset = result->frame_offset;

  // Get the context graph from the stream (may be nullptr)
  const ContextGraph *context_graph = nullptr;
  if (stream != nullptr && stream->GetContextGraph() != nullptr) {
    context_graph = stream->GetContextGraph().get();
  }

  for (int32_t t = 0; t != num_frames; ++t) {
    prev = cur.Vec();
    cur.Clear();

    auto log_probs =
        GetJoinerOutLogSoftmax(model_, p_encoder_out, decoder_out);
    p_encoder_out += encoder_out_dim;

    for (int32_t i = 0; i != static_cast<int32_t>(prev.size()); ++i) {
      auto log_prob = prev[i].log_prob;
      for (auto &p : log_probs[i]) {
        p += log_prob;
      }
    }

    auto topk = TopkIndex(log_probs, max_active_paths_);
    for (auto k : topk) {
      int32_t hyp_index = k / vocab_size;
      int32_t new_token = k % vocab_size;

      Hypothesis new_hyp = prev[hyp_index];
      float context_score = 0;
      new_hyp.log_prob = log_probs[hyp_index][new_token];

      // blank is hardcoded to 0
      // also, it treats unk as blank
      if (new_token != 0 && new_token != unk_id_) {
        new_hyp.ys.push_back(new_token);
        new_hyp.timestamps.push_back(t + frame_offset);
        new_hyp.num_trailing_blanks = 0;

        // Hotword context graph scoring
        if (context_graph != nullptr) {
          auto context_res = context_graph->ForwardOneStep(
              new_hyp.context_state, new_token, false /*strict mode*/);
          context_score = std::get<0>(context_res);
          new_hyp.context_state = std::get<1>(context_res);
          new_hyp.context_scores.push_back(context_score);
        }
      } else {
        ++new_hyp.num_trailing_blanks;
      }

      new_hyp.log_prob += context_score;
      cur.Add(std::move(new_hyp));
    }

    decoder_out = GetDecoderOut(model_, cur);
  }

  result->hyps = std::move(cur);
  result->frame_offset += num_frames;
  result->previous_decoder_out2 = std::move(decoder_out);
}

}  // namespace sherpa_onnx
