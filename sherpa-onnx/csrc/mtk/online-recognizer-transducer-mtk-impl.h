// sherpa-onnx/csrc/mtk/online-recognizer-transducer-mtk-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_MTK_ONLINE_RECOGNIZER_TRANSDUCER_MTK_IMPL_H_
#define SHERPA_ONNX_CSRC_MTK_ONLINE_RECOGNIZER_TRANSDUCER_MTK_IMPL_H_

#include <algorithm>
#include <fstream>
#include <memory>
#include <regex>  // NOLINT
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/context-graph.h"
#include "sherpa-onnx/csrc/endpoint.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/mtk/online-stream-mtk.h"
#include "sherpa-onnx/csrc/mtk/online-transducer-decoder-mtk.h"
#include "sherpa-onnx/csrc/mtk/online-transducer-greedy-search-decoder-mtk.h"
#include "sherpa-onnx/csrc/mtk/online-transducer-modified-beam-search-decoder-mtk.h"
#include "sherpa-onnx/csrc/mtk/online-zipformer-transducer-model-mtk.h"
#include "sherpa-onnx/csrc/online-recognizer-impl.h"
#include "sherpa-onnx/csrc/online-recognizer.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/utils.h"
#include "ssentencepiece/csrc/ssentencepiece.h"

namespace sherpa_onnx {

static OnlineRecognizerResult ConvertMtkResult(
    const OnlineTransducerDecoderResultMtk &src, const SymbolTable &sym_table,
    float frame_shift_ms, int32_t subsampling_factor, int32_t segment,
    int32_t frames_since_start) {
  OnlineRecognizerResult r;
  r.tokens.reserve(src.tokens.size());
  r.timestamps.reserve(src.tokens.size());

  std::string text;
  for (auto i : src.tokens) {
    auto sym = sym_table[i];

    text.append(sym);

    if (sym.size() == 1 && (sym[0] < 0x20 || sym[0] > 0x7e)) {
      std::ostringstream os;
      os << "<0x" << std::hex << std::uppercase
         << (static_cast<int32_t>(sym[0]) & 0xff) << ">";
      sym = os.str();
    }

    r.tokens.push_back(std::move(sym));
  }

  if (sym_table.IsByteBpe()) {
    text = sym_table.DecodeByteBpe(text);
  }

  r.text = std::move(text);

  float frame_shift_s = frame_shift_ms / 1000. * subsampling_factor;
  for (auto t : src.timestamps) {
    float time = frame_shift_s * t;
    r.timestamps.push_back(time);
  }

  r.segment = segment;
  r.start_time = frames_since_start * frame_shift_ms / 1000.;

  return r;
}

class OnlineRecognizerTransducerMtkImpl : public OnlineRecognizerImpl {
 public:
  explicit OnlineRecognizerTransducerMtkImpl(
      const OnlineRecognizerConfig &config)
      : OnlineRecognizerImpl(config),
        config_(config),
        endpoint_(config_.endpoint_config),
        model_(std::make_unique<OnlineZipformerTransducerModelMtk>(
            config.model_config)) {
    if (!config.model_config.tokens_buf.empty()) {
      sym_ = SymbolTable(config.model_config.tokens_buf, false);
    } else {
      sym_ = SymbolTable(config.model_config.tokens, true);
    }

    if (sym_.Contains("<unk>")) {
      unk_id_ = sym_["<unk>"];
    }

    if (config.decoding_method == "modified_beam_search") {
      if (!config_.model_config.bpe_vocab.empty()) {
        bpe_encoder_ = std::make_unique<ssentencepiece::Ssentencepiece>(
            config_.model_config.bpe_vocab);
      }

      if (!config_.hotwords_buf.empty()) {
        InitHotwordsFromBufStr();
      } else if (!config_.hotwords_file.empty()) {
        InitHotwords();
      }

      decoder_ =
          std::make_unique<OnlineTransducerModifiedBeamSearchDecoderMtk>(
              model_.get(), config.max_active_paths, unk_id_,
              config_.blank_penalty);
    } else if (config.decoding_method == "greedy_search") {
      decoder_ = std::make_unique<OnlineTransducerGreedySearchDecoderMtk>(
          model_.get(), unk_id_, config_.blank_penalty);
    } else {
      SHERPA_ONNX_LOGE(
          "Invalid decoding method: '%s'. Support only greedy_search and "
          "modified_beam_search.",
          config.decoding_method.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
  }

#if __ANDROID_API__ >= 9
  template <typename Manager>
  explicit OnlineRecognizerTransducerMtkImpl(
      Manager *mgr, const OnlineRecognizerConfig &config)
      : OnlineRecognizerImpl(mgr, config),
        config_(config),
        endpoint_(config_.endpoint_config),
        model_(std::make_unique<OnlineZipformerTransducerModelMtk>(
            mgr, config_.model_config)) {
    if (!config.model_config.tokens_buf.empty()) {
      sym_ = SymbolTable(config.model_config.tokens_buf, false);
    } else {
      sym_ = SymbolTable(mgr, config.model_config.tokens);
    }

    if (sym_.Contains("<unk>")) {
      unk_id_ = sym_["<unk>"];
    }

    if (config.decoding_method == "modified_beam_search") {
      if (!config_.hotwords_buf.empty()) {
        InitHotwordsFromBufStr();
      } else if (!config_.hotwords_file.empty()) {
        InitHotwords();
      }

      decoder_ =
          std::make_unique<OnlineTransducerModifiedBeamSearchDecoderMtk>(
              model_.get(), config.max_active_paths, unk_id_,
              config_.blank_penalty);
    } else if (config.decoding_method == "greedy_search") {
      decoder_ = std::make_unique<OnlineTransducerGreedySearchDecoderMtk>(
          model_.get(), unk_id_, config_.blank_penalty);
    } else {
      SHERPA_ONNX_LOGE("Invalid decoding method: '%s'.",
                       config.decoding_method.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
  }
#endif

  std::unique_ptr<OnlineStream> CreateStream() const override {
    auto stream =
        std::make_unique<OnlineStreamMtk>(config_.feat_config, hotwords_graph_);
    InitOnlineStream(stream.get());
    return stream;
  }

  std::unique_ptr<OnlineStream> CreateStream(
      const std::string &hotwords) const override {
    auto hws = std::regex_replace(hotwords, std::regex("/"), "\n");
    std::istringstream is(hws);
    std::vector<std::vector<int32_t>> current;
    std::vector<float> current_scores;
    if (!EncodeHotwords(is, config_.model_config.modeling_unit, sym_,
                        bpe_encoder_.get(), &current, &current_scores)) {
      SHERPA_ONNX_LOGE("Encode hotwords failed, skipping, hotwords are: %s",
                       hotwords.c_str());
    }

    int32_t num_default_hws = hotwords_.size();
    int32_t num_hws = current.size();

    current.insert(current.end(), hotwords_.begin(), hotwords_.end());

    if (!current_scores.empty() && !boost_scores_.empty()) {
      current_scores.insert(current_scores.end(), boost_scores_.begin(),
                            boost_scores_.end());
    } else if (!current_scores.empty() && boost_scores_.empty()) {
      current_scores.insert(current_scores.end(), num_default_hws,
                            config_.hotwords_score);
    } else if (current_scores.empty() && !boost_scores_.empty()) {
      current_scores.insert(current_scores.end(), num_hws,
                            config_.hotwords_score);
      current_scores.insert(current_scores.end(), boost_scores_.begin(),
                            boost_scores_.end());
    }

    auto context_graph = std::make_shared<ContextGraph>(
        current, config_.hotwords_score, current_scores);
    auto stream =
        std::make_unique<OnlineStreamMtk>(config_.feat_config, context_graph);
    InitOnlineStream(stream.get());
    return stream;
  }

  bool IsReady(OnlineStream *s) const override {
    return s->GetNumProcessedFrames() + model_->ChunkSize() <
           s->NumFramesReady();
  }

  void DecodeStreams(OnlineStream **ss, int32_t n) const override {
    for (int32_t i = 0; i < n; ++i) {
      DecodeStream(reinterpret_cast<OnlineStreamMtk *>(ss[i]));
    }
  }

  OnlineRecognizerResult GetResult(OnlineStream *s) const override {
    auto *mtk_s = reinterpret_cast<OnlineStreamMtk *>(s);
    OnlineTransducerDecoderResultMtk decoder_result = mtk_s->GetResult();
    decoder_->StripLeadingBlanks(&decoder_result);

    int32_t frame_shift_ms = 10;
    int32_t subsampling_factor = 4;
    auto r =
        ConvertMtkResult(decoder_result, sym_, frame_shift_ms,
                         subsampling_factor, s->GetCurrentSegment(),
                         s->GetNumFramesSinceStart());
    r.text = ApplyInverseTextNormalization(std::move(r.text));
    r.text = ApplyHomophoneReplacer(std::move(r.text));
    return r;
  }

  bool IsEndpoint(OnlineStream *s) const override {
    if (!config_.enable_endpoint) {
      return false;
    }

    int32_t num_processed_frames = s->GetNumProcessedFrames();
    float frame_shift_in_seconds = 0.01;

    int32_t trailing_silence_frames =
        reinterpret_cast<OnlineStreamMtk *>(s)->GetResult().num_trailing_blanks *
        4;

    return endpoint_.IsEndpoint(num_processed_frames, trailing_silence_frames,
                                frame_shift_in_seconds);
  }

  void Reset(OnlineStream *s) const override {
    int32_t context_size = model_->ContextSize();
    auto *mtk_s = reinterpret_cast<OnlineStreamMtk *>(s);

    {
      const auto &r = mtk_s->GetResult();
      if (!r.tokens.empty() && r.tokens.back() != 0 &&
          r.tokens.size() > static_cast<size_t>(context_size)) {
        s->GetCurrentSegment() += 1;
      }
    }

    auto r = decoder_->GetEmptyResult();
    auto last_result = mtk_s->GetResult();

    if (static_cast<int32_t>(last_result.tokens.size()) > context_size) {
      r.tokens = {last_result.tokens.end() - context_size,
                  last_result.tokens.end()};
    }

    // Reset context graph states to root for beam search
    if (config_.decoding_method == "modified_beam_search" &&
        nullptr != s->GetContextGraph()) {
      for (auto it = r.hyps.begin(); it != r.hyps.end(); ++it) {
        it->second.context_state = s->GetContextGraph()->Root();
      }
    }

    mtk_s->SetResult(std::move(r));
    s->Reset();
  }

 private:
  void DecodeStream(OnlineStreamMtk *s) const {
    int32_t chunk_size = model_->ChunkSize();
    int32_t chunk_shift = model_->ChunkShift();

    const auto num_processed_frames = s->GetNumProcessedFrames();

    std::vector<float> features =
        s->GetFrames(num_processed_frames, chunk_size);
    s->GetNumProcessedFrames() += chunk_shift;

    auto &states = s->GetEncoderStates();

    auto p = model_->RunEncoder(features, std::move(states));
    states = std::move(p.second);

    auto &r = s->GetResult();

    // Use the version with hotword support if using beam search
    if (config_.decoding_method == "modified_beam_search") {
      auto *beam_decoder =
          static_cast<OnlineTransducerModifiedBeamSearchDecoderMtk *>(
              decoder_.get());
      beam_decoder->Decode(std::move(p.first), &r, s);
    } else {
      decoder_->Decode(std::move(p.first), &r);
    }
  }

  void InitOnlineStream(OnlineStreamMtk *stream) const {
    auto r = decoder_->GetEmptyResult();

    if (config_.decoding_method == "modified_beam_search" &&
        nullptr != stream->GetContextGraph()) {
      for (auto it = r.hyps.begin(); it != r.hyps.end(); ++it) {
        it->second.context_state = stream->GetContextGraph()->Root();
      }
    }

    stream->SetResult(std::move(r));
    stream->SetEncoderStates(model_->GetEncoderInitStates());
  }

  void InitHotwords() {
    std::ifstream is(config_.hotwords_file);
    if (!is) {
      SHERPA_ONNX_LOGE("Open hotwords file failed: %s",
                       config_.hotwords_file.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    if (!EncodeHotwords(is, config_.model_config.modeling_unit, sym_,
                        bpe_encoder_.get(), &hotwords_, &boost_scores_)) {
      SHERPA_ONNX_LOGE(
          "Failed to encode some hotwords, skip them already, see logs above "
          "for details.");
    }
    hotwords_graph_ = std::make_shared<ContextGraph>(
        hotwords_, config_.hotwords_score, boost_scores_);
  }

  void InitHotwordsFromBufStr() {
    std::istringstream iss(config_.hotwords_buf);
    if (!EncodeHotwords(iss, config_.model_config.modeling_unit, sym_,
                        bpe_encoder_.get(), &hotwords_, &boost_scores_)) {
      SHERPA_ONNX_LOGE(
          "Failed to encode some hotwords, skip them already, see logs above "
          "for details.");
    }
    hotwords_graph_ = std::make_shared<ContextGraph>(
        hotwords_, config_.hotwords_score, boost_scores_);
  }

 private:
  OnlineRecognizerConfig config_;
  SymbolTable sym_;
  Endpoint endpoint_;
  int32_t unk_id_ = -1;

  std::vector<std::vector<int32_t>> hotwords_;
  std::vector<float> boost_scores_;
  ContextGraphPtr hotwords_graph_;
  std::unique_ptr<ssentencepiece::Ssentencepiece> bpe_encoder_;

  std::unique_ptr<OnlineZipformerTransducerModelMtk> model_;
  std::unique_ptr<OnlineTransducerDecoderMtk> decoder_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_MTK_ONLINE_RECOGNIZER_TRANSDUCER_MTK_IMPL_H_
