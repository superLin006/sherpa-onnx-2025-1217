// chattts-sophon-cxx-api.cc
//
// Upper-layer style test: text -> wav with ChatTTS on the Sophon BM1684X TPU,
// using ONLY the public cxx-api (#include "sherpa-onnx/c-api/cxx-api.h").
//
// Usage:
//   chattts-sophon-cxx-api <models_dir> <out.wav> "<text>" [--stream]

#include <cstdint>
#include <cstdio>
#include <string>

#include "sherpa-onnx/c-api/cxx-api.h"

// Streaming progress callback. Prints each chunk + progress and keeps going.
static int32_t ProgressCallback(const float * /*samples*/, int32_t num_samples,
                                float progress, void * /*arg*/) {
  fprintf(stderr, "  [stream] %d samples, progress=%.1f%%\n", num_samples,
          progress * 100);
  return 1;  // continue
}

int32_t main(int32_t argc, char *argv[]) {
  if (argc < 4) {
    fprintf(stderr, "Usage: %s <models_dir> <out.wav> \"<text>\" [--stream]\n",
            argv[0]);
    return 1;
  }
  std::string dir = argv[1];
  std::string out = argv[2];
  std::string text = argv[3];
  bool stream = (argc > 4 && std::string(argv[4]) == "--stream");

  sherpa_onnx::cxx::OfflineTtsConfig config;
  config.model.chattts.gpt = dir + "/chattts-llama_int4_1dev_1024_bm1684x.bmodel";
  config.model.chattts.decoder = dir + "/decoder_1-768-1024_bm1684x.bmodel";
  config.model.chattts.vocos = dir + "/vocos_1-100-2048_bm1684x.bmodel";
  config.model.chattts.vocab = dir + "/asset/tokenizer/vocab.txt";
  config.model.chattts.homophones_map = dir + "/asset/homophones_map.json";
  config.model.chattts.tpu_id = 0;
  config.model.num_threads = 1;
  config.model.debug = false;

  fprintf(stderr, "[INFO] creating ChatTTS (cxx-api)...\n");
  sherpa_onnx::cxx::OfflineTts tts = sherpa_onnx::cxx::OfflineTts::Create(config);
  if (!tts.Get()) {
    fprintf(stderr, "[ERROR] failed to create OfflineTts\n");
    return 1;
  }

  sherpa_onnx::cxx::GeneratedAudio audio;
  if (stream) {
    fprintf(stderr, "[INFO] streaming mode\n");
    audio = tts.Generate(text, 0, 1.0, ProgressCallback, nullptr);
  } else {
    fprintf(stderr, "[INFO] non-streaming mode\n");
    audio = tts.Generate(text);
  }

  if (audio.samples.empty()) {
    fprintf(stderr, "[ERROR] no audio generated\n");
    return 1;
  }

  sherpa_onnx::cxx::WriteWave(out, {audio.samples, audio.sample_rate});
  fprintf(stderr, "[INFO] %d samples @ %d Hz (%.2fs) -> %s\n",
          static_cast<int32_t>(audio.samples.size()), audio.sample_rate,
          audio.samples.size() / static_cast<float>(audio.sample_rate),
          out.c_str());
  return 0;
}
