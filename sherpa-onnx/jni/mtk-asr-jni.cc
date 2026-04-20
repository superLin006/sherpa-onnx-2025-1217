// sherpa-onnx/jni/mtk-asr-jni.cc
//
// Dedicated JNI wrapper for com.xbh.usbcloneclient.MtkAsrWrapper.
//
// Exposes a simple handle-based streaming-ASR API that internally drives
// sherpa-onnx's OnlineRecognizer with provider=mtk (zipformer + hotwords).
//
// Copyright (c)  2026  Xiaomi Corporation

#include <jni.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-recognizer.h"

namespace {

struct MtkAsrSession {
  std::unique_ptr<sherpa_onnx::OnlineRecognizer> recognizer;
  std::unique_ptr<sherpa_onnx::OnlineStream> stream;
  std::mutex mu;  // guards recognizer/stream against concurrent feed+decode
};

static std::string JStringToStd(JNIEnv *env, jstring s) {
  if (s == nullptr) return {};
  const char *p = env->GetStringUTFChars(s, nullptr);
  if (!p) return {};
  std::string out(p);
  env->ReleaseStringUTFChars(s, p);
  return out;
}

static bool FileExists(const std::string &path) {
  if (path.empty()) return false;
  FILE *f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  std::fclose(f);
  return true;
}

}  // namespace

extern "C" {

// long init(String modelDir,
//           String hotwordsFile,
//           String decodingMethod,
//           float  hotwordsScore,
//           int    maxActivePaths)
//
// Expected files in modelDir:
//   encoder.dla
//   decoder_npu.dla
//   joiner.dla
//   decoder_embedding_weight.npy
//   vocab.txt
//
// Returns a non-zero opaque handle on success, 0 on failure.
JNIEXPORT jlong JNICALL
Java_com_xbh_usbcloneclient_MtkAsrWrapper_init(JNIEnv *env, jclass /*cls*/,
                                               jstring modelDir,
                                               jstring hotwordsFile,
                                               jstring decodingMethod,
                                               jfloat hotwordsScore,
                                               jint maxActivePaths) {
  std::string dir = JStringToStd(env, modelDir);
  if (dir.empty()) {
    SHERPA_ONNX_LOGE("modelDir is empty");
    return 0;
  }
  if (dir.back() != '/') dir.push_back('/');

  sherpa_onnx::OnlineRecognizerConfig config;
  config.model_config.transducer.encoder = dir + "encoder.dla";
  config.model_config.transducer.decoder = dir + "decoder_npu.dla";
  config.model_config.transducer.joiner = dir + "joiner.dla";
  config.model_config.provider_config.mtk_decoder_embedding =
      dir + "decoder_embedding_weight.npy";
  config.model_config.tokens = dir + "vocab.txt";
  config.model_config.provider_config.provider = "mtk";
  config.model_config.num_threads = 1;
  config.model_config.debug = false;

  config.feat_config.sampling_rate = 16000;
  config.feat_config.feature_dim = 80;

  std::string method = JStringToStd(env, decodingMethod);
  if (method.empty()) method = "greedy_search";
  config.decoding_method = method;
  config.max_active_paths = maxActivePaths > 0 ? maxActivePaths : 4;

  std::string hw = JStringToStd(env, hotwordsFile);
  if (!hw.empty() && FileExists(hw)) {
    config.hotwords_file = hw;
    config.hotwords_score = hotwordsScore > 0 ? hotwordsScore : 1.5f;
  }

  config.enable_endpoint = true;

  SHERPA_ONNX_LOGE("[MtkAsr] init with dir=%s method=%s hotwords=%s",
                   dir.c_str(), method.c_str(),
                   config.hotwords_file.c_str());

  auto session = std::make_unique<MtkAsrSession>();
  try {
    session->recognizer =
        std::make_unique<sherpa_onnx::OnlineRecognizer>(config);
  } catch (const std::exception &e) {
    SHERPA_ONNX_LOGE("[MtkAsr] failed to construct recognizer: %s", e.what());
    return 0;
  } catch (...) {
    SHERPA_ONNX_LOGE("[MtkAsr] failed to construct recognizer (unknown)");
    return 0;
  }
  if (!session->recognizer) {
    SHERPA_ONNX_LOGE("[MtkAsr] recognizer is null");
    return 0;
  }

  session->stream = session->recognizer->CreateStream();
  if (!session->stream) {
    SHERPA_ONNX_LOGE("[MtkAsr] createStream returned null");
    return 0;
  }

  return reinterpret_cast<jlong>(session.release());
}

// void acceptWaveform(long handle, float[] samples, int sampleRate)
JNIEXPORT void JNICALL
Java_com_xbh_usbcloneclient_MtkAsrWrapper_acceptWaveform(
    JNIEnv *env, jclass /*cls*/, jlong handle, jfloatArray samples,
    jint sampleRate) {
  auto *s = reinterpret_cast<MtkAsrSession *>(handle);
  if (!s || !s->stream || samples == nullptr) return;

  jsize n = env->GetArrayLength(samples);
  if (n <= 0) return;
  std::vector<float> buf(n);
  env->GetFloatArrayRegion(samples, 0, n, buf.data());

  std::lock_guard<std::mutex> lock(s->mu);
  s->stream->AcceptWaveform(sampleRate, buf.data(),
                            static_cast<int32_t>(buf.size()));

  while (s->recognizer->IsReady(s->stream.get())) {
    s->recognizer->DecodeStream(s->stream.get());
  }
}

// Signal input finished — append tail padding so the last chunk decodes.
JNIEXPORT void JNICALL
Java_com_xbh_usbcloneclient_MtkAsrWrapper_inputFinished(JNIEnv * /*env*/,
                                                        jclass /*cls*/,
                                                        jlong handle) {
  auto *s = reinterpret_cast<MtkAsrSession *>(handle);
  if (!s || !s->stream) return;

  std::lock_guard<std::mutex> lock(s->mu);
  s->stream->InputFinished();
  while (s->recognizer->IsReady(s->stream.get())) {
    s->recognizer->DecodeStream(s->stream.get());
  }
}

// String getResult(long handle) — returns the latest decoded text.
JNIEXPORT jstring JNICALL
Java_com_xbh_usbcloneclient_MtkAsrWrapper_getResult(JNIEnv *env,
                                                    jclass /*cls*/,
                                                    jlong handle) {
  auto *s = reinterpret_cast<MtkAsrSession *>(handle);
  if (!s || !s->stream) return env->NewStringUTF("");

  std::lock_guard<std::mutex> lock(s->mu);
  auto result = s->recognizer->GetResult(s->stream.get());
  return env->NewStringUTF(result.text.c_str());
}

// boolean isEndpoint(long handle)
JNIEXPORT jboolean JNICALL
Java_com_xbh_usbcloneclient_MtkAsrWrapper_isEndpoint(JNIEnv * /*env*/,
                                                     jclass /*cls*/,
                                                     jlong handle) {
  auto *s = reinterpret_cast<MtkAsrSession *>(handle);
  if (!s || !s->stream) return JNI_FALSE;
  std::lock_guard<std::mutex> lock(s->mu);
  return s->recognizer->IsEndpoint(s->stream.get()) ? JNI_TRUE : JNI_FALSE;
}

// Reset decoder state after an endpoint (call between utterances).
JNIEXPORT void JNICALL
Java_com_xbh_usbcloneclient_MtkAsrWrapper_reset(JNIEnv * /*env*/,
                                                jclass /*cls*/, jlong handle) {
  auto *s = reinterpret_cast<MtkAsrSession *>(handle);
  if (!s || !s->stream) return;
  std::lock_guard<std::mutex> lock(s->mu);
  s->recognizer->Reset(s->stream.get());
}

// Drop the current stream and create a fresh one with (optional) per-session
// hotwords. Pass empty string to clear.
JNIEXPORT void JNICALL
Java_com_xbh_usbcloneclient_MtkAsrWrapper_restart(JNIEnv *env, jclass /*cls*/,
                                                  jlong handle,
                                                  jstring hotwordsText) {
  auto *s = reinterpret_cast<MtkAsrSession *>(handle);
  if (!s || !s->recognizer) return;

  std::string hw = JStringToStd(env, hotwordsText);
  std::lock_guard<std::mutex> lock(s->mu);
  s->stream.reset();
  s->stream = s->recognizer->CreateStream(hw);
}

// release(long handle) — frees all native resources. Handle invalid after.
JNIEXPORT void JNICALL
Java_com_xbh_usbcloneclient_MtkAsrWrapper_release(JNIEnv * /*env*/,
                                                  jclass /*cls*/,
                                                  jlong handle) {
  auto *s = reinterpret_cast<MtkAsrSession *>(handle);
  if (!s) return;
  delete s;
}

}  // extern "C"
