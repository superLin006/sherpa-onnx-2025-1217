// sherpa-onnx/csrc/sophon/offline-sense-voice-model-sophon.cc
//
// Copyright (c)  2024  Xiaomi Corporation
// Copyright (c)  2026  Sophon BM1684X backend

#include "sherpa-onnx/csrc/sophon/offline-sense-voice-model-sophon.h"

#include <algorithm>
#include <cstring>
#include <mutex>  // NOLINT
#include <string>
#include <utility>
#include <vector>

#include "bmlib_runtime.h"
#include "bmruntime_interface.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

// The exported SenseVoice bmodel takes a single, fixed-shape input that has
// already had LFR applied: [1, num_input_frames, 80 * lfr_window_size] and
// produces logits of shape [1, num_output_frames, vocab_size]. The 4 leading
// "prompt" frames (language / event / itn) are baked into the exported graph,
// so language / text_norm are not fed at runtime; they are kept in the Run()
// signature only to satisfy the recognizer interface.
class OfflineSenseVoiceModelSophon::Impl {
 public:
  ~Impl() {
    if (runtime_) {
      bmrt_destroy(runtime_);
      runtime_ = nullptr;
    }
    if (bm_handle_) {
      bm_dev_free(bm_handle_);
      bm_handle_ = nullptr;
    }
  }

  explicit Impl(const OfflineModelConfig& config) : config_(config) {
    Init(config_.sense_voice.model);
  }

#if __ANDROID_API__ >= 9
  Impl(AAssetManager* /*mgr*/, const OfflineModelConfig& config)
      : config_(config) {
    // Sophon TPU loads the bmodel from a filesystem path; Android assets are
    // not supported. The caller must pass a real path in sense_voice.model.
    Init(config_.sense_voice.model);
  }
#endif

#if __OHOS__
  Impl(NativeResourceManager* /*mgr*/, const OfflineModelConfig& config)
      : config_(config) {
    Init(config_.sense_voice.model);
  }
#endif

  const OfflineSenseVoiceModelMetaData& GetModelMetadata() const {
    return meta_data_;
  }

  std::vector<float> Run(std::vector<float> features, int32_t /*language*/,
                         int32_t /*text_norm*/) {
    // features: (num_frames, 80) raw fbank -> LFR + pad/truncate to the fixed
    // [num_input_frames_, feat_dim_] the bmodel expects.
    features = ApplyLFR(std::move(features));
    int32_t num_lfr_frames =
        static_cast<int32_t>(features.size()) / kInputFeatDim;

    std::vector<float> input(num_input_frames_ * feat_dim_, 0.0f);
    int32_t frames_to_copy = std::min(num_lfr_frames, num_input_frames_);
    if (frames_to_copy > 0) {
      std::memcpy(input.data(), features.data(),
                  frames_to_copy * feat_dim_ * sizeof(float));
    }

    std::lock_guard<std::mutex> lock(mutex_);

    bm_tensor_t input_tensor;
    input_tensor.dtype = BM_FLOAT32;
    input_tensor.st_mode = BM_STORE_1N;
    input_tensor.shape = net_info_->stages[0].input_shapes[0];

    if (bm_malloc_device_byte(bm_handle_, &input_tensor.device_mem,
                              input.size() * sizeof(float)) != BM_SUCCESS) {
      SHERPA_ONNX_LOGE("Sophon: failed to allocate input device memory");
      return {};
    }
    bm_memcpy_s2d(bm_handle_, input_tensor.device_mem, input.data());

    bm_tensor_t output_tensor;
    output_tensor.dtype = BM_FLOAT32;
    output_tensor.st_mode = BM_STORE_1N;
    output_tensor.shape = net_info_->stages[0].output_shapes[0];

    std::vector<float> logits(num_output_frames_ * vocab_size_);
    if (bm_malloc_device_byte(bm_handle_, &output_tensor.device_mem,
                              logits.size() * sizeof(float)) != BM_SUCCESS) {
      SHERPA_ONNX_LOGE("Sophon: failed to allocate output device memory");
      bm_free_device(bm_handle_, input_tensor.device_mem);
      return {};
    }

    bool ok = bmrt_launch_tensor_ex(runtime_, net_info_->name, &input_tensor, 1,
                                    &output_tensor, 1,
                                    /*user_mem=*/true, /*user_stmode=*/false);
    if (!ok) {
      SHERPA_ONNX_LOGE("Sophon: bmrt_launch_tensor_ex failed");
      bm_free_device(bm_handle_, input_tensor.device_mem);
      bm_free_device(bm_handle_, output_tensor.device_mem);
      return {};
    }
    bm_thread_sync(bm_handle_);

    bm_memcpy_d2s(bm_handle_, logits.data(), output_tensor.device_mem);

    bm_free_device(bm_handle_, input_tensor.device_mem);
    bm_free_device(bm_handle_, output_tensor.device_mem);

    return logits;
  }

 private:
  void Init(const std::string& bmodel_path) {
    if (bm_dev_request(&bm_handle_, config_.sense_voice.tpu_id) != BM_SUCCESS) {
      SHERPA_ONNX_LOGE("Sophon: bm_dev_request failed (tpu_id=%d)",
                       config_.sense_voice.tpu_id);
      SHERPA_ONNX_EXIT(-1);
    }

    runtime_ = bmrt_create(bm_handle_);
    if (!runtime_) {
      SHERPA_ONNX_LOGE("Sophon: bmrt_create failed");
      SHERPA_ONNX_EXIT(-1);
    }

    if (!bmrt_load_bmodel(runtime_, bmodel_path.c_str())) {
      SHERPA_ONNX_LOGE("Sophon: failed to load bmodel '%s'",
                       bmodel_path.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    const char** net_names = nullptr;
    int num_nets = bmrt_get_network_number(runtime_);
    bmrt_get_network_names(runtime_, &net_names);
    if (num_nets <= 0 || !net_names) {
      SHERPA_ONNX_LOGE("Sophon: bmodel contains no network");
      SHERPA_ONNX_EXIT(-1);
    }
    net_info_ = bmrt_get_network_info(runtime_, net_names[0]);
    SHERPA_ONNX_LOGE("Sophon: using network '%s' (input_num=%d output_num=%d)",
                     net_names[0], net_info_->input_num, net_info_->output_num);
    free(net_names);

    if (!net_info_) {
      SHERPA_ONNX_LOGE("Sophon: bmrt_get_network_info failed");
      SHERPA_ONNX_EXIT(-1);
    }

    // Derive shapes from the bmodel so a re-export with a different length
    // keeps working.
    const bm_shape_t& in = net_info_->stages[0].input_shapes[0];
    const bm_shape_t& out = net_info_->stages[0].output_shapes[0];
    num_input_frames_ = in.dims[in.num_dims - 2];
    feat_dim_ = in.dims[in.num_dims - 1];
    num_output_frames_ = out.dims[out.num_dims - 2];
    vocab_size_ = out.dims[out.num_dims - 1];

    InitMetaData();

    SHERPA_ONNX_LOGE("Sophon SenseVoice: in=[%d,%d] out=[%d,%d] lfr=%d/%d",
                     num_input_frames_, feat_dim_, num_output_frames_,
                     vocab_size_, meta_data_.window_size,
                     meta_data_.window_shift);
  }

  void InitMetaData() {
    meta_data_.with_itn_id = 14;
    meta_data_.without_itn_id = 15;
    meta_data_.window_size = 7;
    meta_data_.window_shift = 6;
    meta_data_.vocab_size = vocab_size_;
    meta_data_.normalize_samples = 0;

    meta_data_.lang2id = {{"auto", 0}, {"zh", 3}, {"en", 4},
                          {"yue", 7},  {"ja", 11}, {"ko", 12}};
  }

  // LFR: stack window_size frames with shift window_shift.
  // Input: [num_frames, 80] -> Output: [lfr_frames, 560]
  std::vector<float> ApplyLFR(std::vector<float> in) const {
    const int32_t feat_dim = 80;
    int32_t num_frames = static_cast<int32_t>(in.size()) / feat_dim;

    if (num_frames < meta_data_.window_size) {
      SHERPA_ONNX_LOGE("Input frames (%d) less than window size (%d)",
                       num_frames, meta_data_.window_size);
      in.resize(meta_data_.window_size * feat_dim, 0.0f);
      num_frames = meta_data_.window_size;
    }

    int32_t out_num_frames =
        (num_frames - meta_data_.window_size) / meta_data_.window_shift + 1;

    std::vector<float> out(out_num_frames * kInputFeatDim);
    for (int32_t i = 0; i < out_num_frames; ++i) {
      int32_t start_frame = i * meta_data_.window_shift;
      for (int32_t j = 0; j < meta_data_.window_size; ++j) {
        std::memcpy(out.data() + i * kInputFeatDim + j * feat_dim,
                    in.data() + (start_frame + j) * feat_dim,
                    feat_dim * sizeof(float));
      }
    }
    return out;
  }

  OfflineModelConfig config_;
  OfflineSenseVoiceModelMetaData meta_data_;

  bm_handle_t bm_handle_ = nullptr;
  void* runtime_ = nullptr;
  const bm_net_info_t* net_info_ = nullptr;
  std::mutex mutex_;

  int32_t num_input_frames_ = kModelInputFrames;
  int32_t feat_dim_ = kInputFeatDim;
  int32_t num_output_frames_ = kOutputFrames;
  int32_t vocab_size_ = kVocabSize;
};

OfflineSenseVoiceModelSophon::~OfflineSenseVoiceModelSophon() = default;

OfflineSenseVoiceModelSophon::OfflineSenseVoiceModelSophon(
    const OfflineModelConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

#if __ANDROID_API__ >= 9
OfflineSenseVoiceModelSophon::OfflineSenseVoiceModelSophon(
    AAssetManager* mgr, const OfflineModelConfig& config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}
#endif

#if __OHOS__
OfflineSenseVoiceModelSophon::OfflineSenseVoiceModelSophon(
    NativeResourceManager* mgr, const OfflineModelConfig& config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}
#endif

std::vector<float> OfflineSenseVoiceModelSophon::Run(std::vector<float> features,
                                                     int32_t language,
                                                     int32_t text_norm) const {
  return impl_->Run(std::move(features), language, text_norm);
}

const OfflineSenseVoiceModelMetaData&
OfflineSenseVoiceModelSophon::GetModelMetadata() const {
  return impl_->GetModelMetadata();
}

}  // namespace sherpa_onnx
