// sense-voice-sophon-pcm-asr.cc
//
// Minimal test: WAV file in -> recognized text out, running SenseVoice on the
// Sophon BM1684X TPU via the sherpa-onnx cxx-api (provider = "sophon").

#include <cstdio>
#include <iostream>
#include <string>

#include "sherpa-onnx/c-api/cxx-api.h"

int main(int argc, char *argv[]) {
  if (argc < 4) {
    fprintf(stderr,
            "Usage: %s <sensevoice.bmodel> <tokens.txt> <test.wav> "
            "[language] [use_itn]\n"
            "  language: auto|zh|en|ja|ko|yue  (default auto)\n"
            "  use_itn : 0|1                   (default 0)\n",
            argv[0]);
    return 1;
  }

  const std::string bmodel = argv[1];
  const std::string tokens = argv[2];
  const std::string wav_path = argv[3];
  const std::string language = argc > 4 ? argv[4] : "auto";
  const bool use_itn = argc > 5 ? (std::string(argv[5]) == "1") : false;

  sherpa_onnx::cxx::Wave wave = sherpa_onnx::cxx::ReadWave(wav_path);
  if (wave.samples.empty()) {
    fprintf(stderr, "[ERROR] cannot read wav: %s\n", wav_path.c_str());
    return 1;
  }
  fprintf(stderr, "[INFO] wav: %d samples @ %d Hz (%.2fs)\n",
          static_cast<int>(wave.samples.size()), wave.sample_rate,
          wave.samples.size() / static_cast<float>(wave.sample_rate));

  sherpa_onnx::cxx::OfflineRecognizerConfig config;
  config.model_config.provider = "sophon";  // <-- selects the BM1684X TPU
  config.model_config.sense_voice.model = bmodel;
  config.model_config.sense_voice.language = language;
  config.model_config.sense_voice.use_itn = use_itn;
  config.model_config.tokens = tokens;
  config.model_config.num_threads = 1;
  config.model_config.debug = true;

  fprintf(stderr, "[INFO] creating recognizer (provider=sophon)...\n");
  sherpa_onnx::cxx::OfflineRecognizer recognizer =
      sherpa_onnx::cxx::OfflineRecognizer::Create(config);
  if (!recognizer.Get()) {
    fprintf(stderr, "[ERROR] failed to create recognizer\n");
    return 1;
  }

  sherpa_onnx::cxx::OfflineStream stream = recognizer.CreateStream();
  stream.AcceptWaveform(wave.sample_rate, wave.samples.data(),
                        static_cast<int>(wave.samples.size()));
  recognizer.Decode(&stream);

  std::string text = recognizer.GetResult(&stream).text;
  fprintf(stderr, "[INFO] done.\n");
  printf("RESULT: %s\n", text.c_str());
  return 0;
}
