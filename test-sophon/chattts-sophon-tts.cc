// chattts-sophon-tts.cc
//
// On-board smoke test: text -> wav, running ChatTTS on the Sophon BM1684X TPU
// via sherpa-onnx's internal OfflineTts API.
//
// Usage:
//   chattts-sophon-tts <models_dir> <out.wav> "<text>" [--stream]
// where <models_dir> contains:
//   chattts-llama_int4_1dev_1024_bm1684x.bmodel
//   decoder_1-768-1024_bm1684x.bmodel
//   vocos_1-100-2048_bm1684x.bmodel
//   asset/tokenizer/vocab.txt
//   asset/homophones_map.json

#include <cstdio>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/offline-tts.h"
#include "sherpa-onnx/csrc/wave-writer.h"

int main(int argc, char *argv[]) {
  if (argc < 4) {
    fprintf(stderr,
            "Usage: %s <models_dir> <out.wav> \"<text>\" [--stream]\n",
            argv[0]);
    return 1;
  }
  std::string dir = argv[1];
  std::string out = argv[2];
  std::string text = argv[3];
  bool stream = (argc > 4 && std::string(argv[4]) == "--stream");

  sherpa_onnx::OfflineTtsConfig config;
  auto &c = config.model.chattts;
  c.gpt = dir + "/chattts-llama_int4_1dev_1024_bm1684x.bmodel";
  c.decoder = dir + "/decoder_1-768-1024_bm1684x.bmodel";
  c.vocos = dir + "/vocos_1-100-2048_bm1684x.bmodel";
  c.vocab = dir + "/asset/tokenizer/vocab.txt";
  c.homophones_map = dir + "/asset/homophones_map.json";
  c.tpu_id = 0;
  config.model.num_threads = 1;
  config.model.debug = true;

  fprintf(stderr, "[INFO] creating ChatTTS OfflineTts...\n");
  sherpa_onnx::OfflineTts tts(config);

  sherpa_onnx::GeneratedAudio audio;
  if (stream) {
    fprintf(stderr, "[INFO] streaming mode\n");
    int32_t chunk_idx = 0;
    audio = tts.Generate(
        text, /*sid=*/0, /*speed=*/1.0,
        [&chunk_idx](const float * /*samples*/, int32_t n, float /*progress*/) {
          fprintf(stderr, "  [stream] chunk %d: %d samples\n", chunk_idx++, n);
          return 1;  // keep going
        });
  } else {
    fprintf(stderr, "[INFO] non-streaming mode\n");
    audio = tts.Generate(text, /*sid=*/0, /*speed=*/1.0);
  }

  if (audio.samples.empty()) {
    fprintf(stderr, "[ERROR] no audio generated\n");
    return 1;
  }

  bool ok = sherpa_onnx::WriteWave(out, audio.sample_rate, audio.samples.data(),
                                   static_cast<int32_t>(audio.samples.size()));
  fprintf(stderr, "[INFO] %d samples @ %d Hz (%.2fs) -> %s (%s)\n",
          static_cast<int32_t>(audio.samples.size()), audio.sample_rate,
          audio.samples.size() / static_cast<float>(audio.sample_rate),
          out.c_str(), ok ? "OK" : "FAILED");
  return ok ? 0 : 1;
}
