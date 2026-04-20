// sherpa-onnx/csrc/mtk/online-zipformer-transducer-model-mtk.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/mtk/online-zipformer-transducer-model-mtk.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/log.h"
#include "sherpa-onnx/csrc/mtk/macros.h"
#include "sherpa-onnx/csrc/mtk/mtk-npu-executor.h"

namespace sherpa_onnx {

// Model constants matching the exported DLA models
static constexpr int32_t kSampleRate = 16000;
static constexpr int32_t kNMels = 80;
static constexpr int32_t kSegment = 103;       // encoder input frames per chunk
static constexpr int32_t kOffset = 96;         // stride per chunk
static constexpr int32_t kEncoderOutT = 24;    // encoder output time steps
static constexpr int32_t kEncDim = 256;        // encoder output dim
static constexpr int32_t kDecDim = 512;        // decoder output dim
static constexpr int32_t kVocabSize = 6254;
static constexpr int32_t kContextSize = 2;
static constexpr int32_t kEncNumInputs = 36;
static constexpr int32_t kEncNumOutputs = 36;

// ---------------------------------------------------------------------------
// NPY file loader (for decoder_embedding_weight.npy)
// ---------------------------------------------------------------------------
static bool LoadNpyFloat32(const std::string &path, std::vector<float> &data,
                           int32_t &rows, int32_t &cols) {
  FILE *fp = fopen(path.c_str(), "rb");
  if (!fp) {
    SHERPA_ONNX_LOGE("Cannot open npy file: %s", path.c_str());
    return false;
  }

  // Read magic + version
  uint8_t magic[6];
  uint8_t version[2];
  if (fread(magic, 1, 6, fp) != 6 || fread(version, 1, 2, fp) != 2) {
    fclose(fp);
    return false;
  }

  if (magic[0] != 0x93 || magic[1] != 'N' || magic[2] != 'U' ||
      magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y') {
    SHERPA_ONNX_LOGE("Not a valid npy file: %s", path.c_str());
    fclose(fp);
    return false;
  }

  uint32_t header_len = 0;
  if (version[0] == 1) {
    uint16_t hlen16;
    if (fread(&hlen16, 2, 1, fp) != 1) {
      fclose(fp);
      return false;
    }
    header_len = hlen16;
  } else {
    uint32_t hlen32;
    if (fread(&hlen32, 4, 1, fp) != 1) {
      fclose(fp);
      return false;
    }
    header_len = hlen32;
  }

  std::vector<char> header_buf(header_len + 1, 0);
  if (fread(header_buf.data(), 1, header_len, fp) != header_len) {
    fclose(fp);
    return false;
  }
  std::string header(header_buf.data());

  // Parse shape from header: look for 'shape': (R, C)
  size_t shape_pos = header.find("'shape'");
  if (shape_pos == std::string::npos) {
    shape_pos = header.find("\"shape\"");
  }
  if (shape_pos == std::string::npos) {
    SHERPA_ONNX_LOGE("Cannot find shape in npy header");
    fclose(fp);
    return false;
  }

  size_t lp = header.find('(', shape_pos);
  size_t rp = header.find(')', lp);
  if (lp == std::string::npos || rp == std::string::npos) {
    fclose(fp);
    return false;
  }

  std::string shape_str = header.substr(lp + 1, rp - lp - 1);
  shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ' '),
                  shape_str.end());

  size_t comma = shape_str.find(',');
  if (comma == std::string::npos) {
    fclose(fp);
    return false;
  }
  rows = std::stoi(shape_str.substr(0, comma));
  std::string col_str = shape_str.substr(comma + 1);
  col_str.erase(std::remove(col_str.begin(), col_str.end(), ','),
                col_str.end());
  if (col_str.empty()) {
    fclose(fp);
    return false;
  }
  cols = std::stoi(col_str);

  size_t n_floats = static_cast<size_t>(rows) * cols;
  data.resize(n_floats);
  size_t nread = fread(data.data(), sizeof(float), n_floats, fp);
  fclose(fp);

  if (nread != n_floats) {
    SHERPA_ONNX_LOGE("npy read error: expected %zu floats, got %zu", n_floats,
                     nread);
    return false;
  }

  SHERPA_ONNX_LOGI("Loaded npy: %s shape=[%d,%d]", path.c_str(), rows, cols);
  return true;
}

// ---------------------------------------------------------------------------
// CPU embedding lookup with mask (for decoder)
// ---------------------------------------------------------------------------
static void EmbedTokens(const int64_t *token_ids, int32_t context_size,
                        const float *emb_weight, int32_t emb_vocab,
                        int32_t emb_dim, float *out) {
  for (int32_t i = 0; i < context_size; ++i) {
    int64_t tid = token_ids[i];
    float mask = (tid >= 0) ? 1.0f : 0.0f;
    int32_t safe_id = (tid >= 0) ? static_cast<int32_t>(tid) : 0;
    safe_id = std::min(safe_id, emb_vocab - 1);
    const float *row = emb_weight + static_cast<size_t>(safe_id) * emb_dim;
    float *dst = out + i * emb_dim;
    for (int32_t d = 0; d < emb_dim; ++d) {
      dst[d] = row[d] * mask;
    }
  }
}

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------
class OnlineZipformerTransducerModelMtk::Impl {
 public:
  explicit Impl(const OnlineModelConfig &config) : config_(config) {
    InitEncoder(config.transducer.encoder);
    InitDecoder(config.transducer.decoder);
    InitJoiner(config.transducer.joiner);
    InitEmbedding(config.provider_config.mtk_decoder_embedding);
  }

#if __ANDROID_API__ >= 9
  Impl(AAssetManager *mgr, const OnlineModelConfig &config) : config_(config) {
    // For Android, load DLA from asset manager to temp files, then init
    // For now, fall back to direct path loading
    InitEncoder(config.transducer.encoder);
    InitDecoder(config.transducer.decoder);
    InitJoiner(config.transducer.joiner);
    InitEmbedding(config.provider_config.mtk_decoder_embedding);
  }
#endif

#if __OHOS__
  Impl(NativeResourceManager *mgr, const OnlineModelConfig &config)
      : config_(config) {
    InitEncoder(config.transducer.encoder);
    InitDecoder(config.transducer.decoder);
    InitJoiner(config.transducer.joiner);
    InitEmbedding(config.provider_config.mtk_decoder_embedding);
  }
#endif

  std::vector<std::vector<float>> GetEncoderInitStates() const {
    EncoderCacheMtk cache;
    return std::move(cache.buf);
  }

  std::pair<std::vector<float>, std::vector<std::vector<float>>> RunEncoder(
      const std::vector<float> &features,
      std::vector<std::vector<float>> states) {
    // Build input tensor buffers (36 total)
    std::vector<MtkTensorBuffer> inputs(kEncNumInputs);

    // Input 0: x [1,103,80]
    inputs[0].data = const_cast<float *>(features.data());
    inputs[0].bytes =
        static_cast<size_t>(1) * kSegment * kNMels * sizeof(float);
    inputs[0].type = MtkTensorDataType::kFloat32;

    // Inputs 1..35: cached states
    for (int32_t i = 0; i < EncoderCacheMtk::kNumCachedStates; ++i) {
      inputs[i + 1].data = states[i].data();
      inputs[i + 1].bytes = states[i].size() * sizeof(float);
      inputs[i + 1].type = MtkTensorDataType::kFloat32;
    }

    // Build output tensor buffers (36 total)
    std::vector<float> encoder_out(
        static_cast<size_t>(1) * kEncoderOutT * kEncDim);
    EncoderCacheMtk new_cache;

    std::vector<MtkTensorBuffer> outputs(kEncNumOutputs);

    // Output 0: encoder_out [1,24,256]
    outputs[0].data = encoder_out.data();
    outputs[0].bytes =
        static_cast<size_t>(1) * kEncoderOutT * kEncDim * sizeof(float);
    outputs[0].type = MtkTensorDataType::kFloat32;

    // Outputs 1..35: new cached states
    for (int32_t i = 0; i < EncoderCacheMtk::kNumCachedStates; ++i) {
      outputs[i + 1].data = new_cache.Data(i);
      outputs[i + 1].bytes = new_cache.Bytes(i);
      outputs[i + 1].type = MtkTensorDataType::kFloat32;
    }

    bool ok = encoder_exec_->RunForMultipleInputsOutputs(inputs, outputs);
    if (!ok) {
      SHERPA_ONNX_LOGE("MTK encoder inference failed");
    }

    return {std::move(encoder_out), std::move(new_cache.buf)};
  }

  std::vector<float> RunDecoder(const std::vector<int64_t> &tokens) {
    // CPU embedding lookup
    std::vector<float> emb_input(
        static_cast<size_t>(kContextSize) * emb_dim_, 0.0f);
    EmbedTokens(tokens.data(), kContextSize, emb_weight_.data(),
                emb_vocab_size_, emb_dim_, emb_input.data());

    std::vector<MtkTensorBuffer> inputs(1);
    inputs[0].data = emb_input.data();
    inputs[0].bytes = emb_input.size() * sizeof(float);
    inputs[0].type = MtkTensorDataType::kFloat32;

    std::vector<float> decoder_out(kDecDim);
    std::vector<MtkTensorBuffer> outputs(1);
    outputs[0].data = decoder_out.data();
    outputs[0].bytes = static_cast<size_t>(kDecDim) * sizeof(float);
    outputs[0].type = MtkTensorDataType::kFloat32;

    bool ok = decoder_exec_->RunForMultipleInputsOutputs(inputs, outputs);
    if (!ok) {
      SHERPA_ONNX_LOGE("MTK decoder inference failed");
    }

    return decoder_out;
  }

  std::vector<float> RunJoiner(const float *encoder_out,
                               const float *decoder_out) {
    std::vector<MtkTensorBuffer> inputs(2);
    inputs[0].data = const_cast<float *>(encoder_out);
    inputs[0].bytes = static_cast<size_t>(kEncDim) * sizeof(float);
    inputs[0].type = MtkTensorDataType::kFloat32;

    inputs[1].data = const_cast<float *>(decoder_out);
    inputs[1].bytes = static_cast<size_t>(kDecDim) * sizeof(float);
    inputs[1].type = MtkTensorDataType::kFloat32;

    std::vector<float> joiner_out(kVocabSize);
    std::vector<MtkTensorBuffer> outputs(1);
    outputs[0].data = joiner_out.data();
    outputs[0].bytes = static_cast<size_t>(kVocabSize) * sizeof(float);
    outputs[0].type = MtkTensorDataType::kFloat32;

    bool ok = joiner_exec_->RunForMultipleInputsOutputs(inputs, outputs);
    if (!ok) {
      SHERPA_ONNX_LOGE("MTK joiner inference failed");
    }

    return joiner_out;
  }

  int32_t ContextSize() const { return kContextSize; }
  int32_t ChunkSize() const { return kSegment; }
  int32_t ChunkShift() const { return kOffset; }
  int32_t VocabSize() const { return kVocabSize; }
  int32_t EncoderOutDim() const { return kEncDim; }
  int32_t EncoderOutFrames() const { return kEncoderOutT; }

 private:
  void InitEncoder(const std::string &model_path) {
    encoder_exec_ = std::make_unique<MtkNpuExecutor>("ZipformerEncoder");

    // Input shapes: [0] = [1,103,80], [1..35] = cached state shapes
    std::vector<std::vector<uint32_t>> input_shapes;
    input_shapes.push_back({1, kSegment, kNMels});
    for (int32_t i = 0; i < EncoderCacheMtk::kNumCachedStates; ++i) {
      input_shapes.push_back(
          {static_cast<uint32_t>(EncoderCacheMtk::kCacheStateSizes[i])});
    }

    // Output shapes: [0] = [1,24,256], [1..35] = same as input cached states
    std::vector<std::vector<uint32_t>> output_shapes;
    output_shapes.push_back({1, kEncoderOutT, kEncDim});
    for (int32_t i = 0; i < EncoderCacheMtk::kNumCachedStates; ++i) {
      output_shapes.push_back(
          {static_cast<uint32_t>(EncoderCacheMtk::kCacheStateSizes[i])});
    }

    bool ok = encoder_exec_->Initialize(model_path, input_shapes, output_shapes,
                                        /*input_type=*/0, /*output_type=*/0);
    if (!ok) {
      SHERPA_ONNX_LOGE("Failed to init MTK encoder: %s", model_path.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
    SHERPA_ONNX_LOGI("MTK Encoder loaded: %s", model_path.c_str());
  }

  void InitDecoder(const std::string &model_path) {
    decoder_exec_ = std::make_unique<MtkNpuExecutor>("ZipformerDecoder");

    std::vector<std::vector<uint32_t>> input_shapes = {
        {1, kContextSize, static_cast<uint32_t>(kDecDim)}};
    std::vector<std::vector<uint32_t>> output_shapes = {
        {1, static_cast<uint32_t>(kDecDim)}};

    bool ok = decoder_exec_->Initialize(model_path, input_shapes, output_shapes,
                                        /*input_type=*/0, /*output_type=*/0);
    if (!ok) {
      SHERPA_ONNX_LOGE("Failed to init MTK decoder: %s", model_path.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
    SHERPA_ONNX_LOGI("MTK Decoder loaded: %s", model_path.c_str());
  }

  void InitJoiner(const std::string &model_path) {
    joiner_exec_ = std::make_unique<MtkNpuExecutor>("ZipformerJoiner");

    std::vector<std::vector<uint32_t>> input_shapes = {
        {1, static_cast<uint32_t>(kEncDim)},
        {1, static_cast<uint32_t>(kDecDim)}};
    std::vector<std::vector<uint32_t>> output_shapes = {
        {1, static_cast<uint32_t>(kVocabSize)}};

    bool ok = joiner_exec_->Initialize(model_path, input_shapes, output_shapes,
                                       /*input_type=*/0, /*output_type=*/0);
    if (!ok) {
      SHERPA_ONNX_LOGE("Failed to init MTK joiner: %s", model_path.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
    SHERPA_ONNX_LOGI("MTK Joiner loaded: %s", model_path.c_str());
  }

  void InitEmbedding(const std::string &emb_path) {
    if (emb_path.empty()) {
      SHERPA_ONNX_LOGE(
          "decoder_embedding path is empty. For MTK zipformer, you must "
          "provide the decoder embedding weight file via "
          "--mtk-decoder-embedding=<path-to-npy>");
      SHERPA_ONNX_EXIT(-1);
    }

    int32_t rows = 0, cols = 0;
    if (!LoadNpyFloat32(emb_path, emb_weight_, rows, cols)) {
      SHERPA_ONNX_LOGE("Failed to load decoder embedding: %s",
                       emb_path.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
    emb_vocab_size_ = rows;
    emb_dim_ = cols;
    SHERPA_ONNX_LOGI("Decoder embedding loaded: vocab=%d dim=%d", rows, cols);
  }

 private:
  OnlineModelConfig config_;
  std::unique_ptr<MtkNpuExecutor> encoder_exec_;
  std::unique_ptr<MtkNpuExecutor> decoder_exec_;
  std::unique_ptr<MtkNpuExecutor> joiner_exec_;

  std::vector<float> emb_weight_;
  int32_t emb_vocab_size_ = 0;
  int32_t emb_dim_ = 0;
};

// constexpr member definition (required before C++17 for ODR-use)
constexpr int32_t EncoderCacheMtk::kCacheStateSizes[];

// ---------------------------------------------------------------------------
// Public interface delegating to Impl
// ---------------------------------------------------------------------------

OnlineZipformerTransducerModelMtk::~OnlineZipformerTransducerModelMtk() =
    default;

OnlineZipformerTransducerModelMtk::OnlineZipformerTransducerModelMtk(
    const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

#if __ANDROID_API__ >= 9
OnlineZipformerTransducerModelMtk::OnlineZipformerTransducerModelMtk(
    AAssetManager *mgr, const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}
#endif

#if __OHOS__
OnlineZipformerTransducerModelMtk::OnlineZipformerTransducerModelMtk(
    NativeResourceManager *mgr, const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}
#endif

std::vector<std::vector<float>>
OnlineZipformerTransducerModelMtk::GetEncoderInitStates() const {
  return impl_->GetEncoderInitStates();
}

std::pair<std::vector<float>, std::vector<std::vector<float>>>
OnlineZipformerTransducerModelMtk::RunEncoder(
    const std::vector<float> &features,
    std::vector<std::vector<float>> states) {
  return impl_->RunEncoder(features, std::move(states));
}

std::vector<float> OnlineZipformerTransducerModelMtk::RunDecoder(
    const std::vector<int64_t> &tokens) {
  return impl_->RunDecoder(tokens);
}

std::vector<float> OnlineZipformerTransducerModelMtk::RunJoiner(
    const float *encoder_out, const float *decoder_out) {
  return impl_->RunJoiner(encoder_out, decoder_out);
}

int32_t OnlineZipformerTransducerModelMtk::ContextSize() const {
  return impl_->ContextSize();
}

int32_t OnlineZipformerTransducerModelMtk::ChunkSize() const {
  return impl_->ChunkSize();
}

int32_t OnlineZipformerTransducerModelMtk::ChunkShift() const {
  return impl_->ChunkShift();
}

int32_t OnlineZipformerTransducerModelMtk::VocabSize() const {
  return impl_->VocabSize();
}

int32_t OnlineZipformerTransducerModelMtk::EncoderOutDim() const {
  return impl_->EncoderOutDim();
}

int32_t OnlineZipformerTransducerModelMtk::EncoderOutFrames() const {
  return impl_->EncoderOutFrames();
}

}  // namespace sherpa_onnx
