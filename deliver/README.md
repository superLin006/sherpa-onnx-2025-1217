# SenseVoice + ChatTTS on Sophon BM1684X — 交付与使用说明

> 在算能 BM1684X TPU 上，通过 **sherpa-onnx** 框架运行：
> - **SenseVoice**：语音识别 ASR（音频 → 文字）
> - **ChatTTS**：语音合成 TTS（文字 → 音频，支持流式）
>
> 两个模型推理都跑在 TPU 上，对外是统一的 sherpa-onnx C++ 接口。

---

## 一、环境要求

| 项 | 要求 |
|---|---|
| 设备 | 算能 BM1684X 板卡（aarch64） |
| 系统 | Ubuntu 20.04（glibc 2.31，本包二进制按此构建） |
| 驱动 | libsophon（板上 `/opt/sophon/libsophon-current/lib`，本包已验证 0.5.1） |
| 访问 | ssh（示例板 `root@172.16.40.75`，端口 22） |

> 本包的 `.so` 与可执行文件是用 Ubuntu 20.04 / GCC 9.4 / glibc 2.31 交叉编译的，
> 最高仅依赖 `GLIBC_2.29`，与板卡系统匹配。换更新/更旧的板卡镜像需重新编译。

---

## 二、交付目录结构

```
deliver/
├── bin/
│   ├── sense-voice-sophon-pcm-asr     # ASR 测试程序（WAV → 文字）
│   └── chattts-sophon-cxx-api         # TTS 测试程序（文字 → WAV）
├── include/
│   └── sherpa-onnx/c-api/
│       ├── cxx-api.h                  # ← 上层应用 #include 这个（C++ 接口）
│       └── c-api.h                    # 底层 C 接口（一般不直接用）
├── lib/
│   ├── libsherpa-onnx-cxx-api.so      # ← 上层应用链接这两个
│   ├── libsherpa-onnx-c-api.so
│   └── libonnxruntime.so
├── models/
│   ├── sensevoice/
│   │   ├── sensevoice_small_F16.bmodel
│   │   └── tokens.txt
│   └── chattts/
│       ├── chattts-llama_int4_1dev_1024_bm1684x.bmodel   # GPT
│       ├── decoder_1-768-1024_bm1684x.bmodel             # DVAE decoder
│       ├── vocos_1-100-2048_bm1684x.bmodel               # Vocos
│       └── asset/
│           ├── tokenizer/vocab.txt
│           └── homophones_map.json
├── test_data/
│   ├── test_zh.wav / test_en.wav      # ASR 测试音频
│   ├── run_asr.sh                     # 一键跑 ASR
│   └── run_tts.sh                     # 一键跑 TTS
└── README.md
```

---

## 三、快速开始（板卡上验证）

### 1. 推送到板卡

```bash
# 在 PC 上（把 deliver/ 整个拷到板卡）
scp -r deliver root@172.16.40.75:/root/sophon_demo
```

### 2. 跑 ASR（语音识别）

```bash
ssh root@172.16.40.75
cd /root/sophon_demo
chmod +x bin/* test_data/*.sh

# 跑全部测试音频
sh test_data/run_asr.sh
# 或指定文件 + 语言
sh test_data/run_asr.sh test_zh.wav zh
```

预期输出：
```
=== ASR: test_zh.wav (lang=zh) ===
RESULT: 对我做了介绍啊，那么我想说的是呢，大家如果对我的研究感兴趣呢。
```

### 3. 跑 TTS（语音合成）

```bash
# 非流式（默认文本）
sh test_data/run_tts.sh

# 自定义文本
sh test_data/run_tts.sh "你好，欢迎使用算能语音合成。"

# 流式（每块带进度回调）
sh test_data/run_tts.sh "流式合成，边生成边输出。" --stream
```

预期输出（流式）：
```
  [stream] 22362 samples, progress=3.5%
  [stream] 12288 samples, progress=4.6%
  ...
  [stream] 12288 samples, progress=100.0%
[INFO] 71514 samples @ 24000 Hz (2.98s) -> .../tts_out.wav
```
合成结果保存为 `test_data/tts_out.wav`（24 kHz 单声道）。

---

## 四、给开发人员：如何在自己的程序里调用

SDK 对外只有头文件 + 两个 `.so`，本包已附带（见 `include/` 与 `lib/`）。链接方式：

```cmake
# 把交付包根目录当作 SDK 根（含 include/ 和 lib/）
set(SDK_DIR /path/to/deliver)
target_include_directories(your_app PRIVATE "${SDK_DIR}/include")
target_link_libraries(your_app PRIVATE
    "${SDK_DIR}/lib/libsherpa-onnx-cxx-api.so"
    "${SDK_DIR}/lib/libsherpa-onnx-c-api.so")
# 运行时确保 LD_LIBRARY_PATH 含 lib/ 和 /opt/sophon/libsophon-current/lib
```

源码里这样引用头文件（注意带 `sherpa-onnx/c-api/` 前缀）：

```cpp
#include "sherpa-onnx/c-api/cxx-api.h"   // C++ 接口（推荐）
```

一行编译示例（aarch64，板卡本地或交叉）：

```bash
g++ -std=c++17 your_app.cc -o your_app \
    -I deliver/include \
    -L deliver/lib -lsherpa-onnx-cxx-api -lsherpa-onnx-c-api \
    -Wl,-rpath,'$ORIGIN/lib'
```

### 4.1 ASR：音频 → 文字

```cpp
#include "sherpa-onnx/c-api/cxx-api.h"
using namespace sherpa_onnx::cxx;

OfflineRecognizerConfig config;
config.model_config.provider               = "sophon";   // ← 选 TPU 后端
config.model_config.sense_voice.model      = "models/sensevoice/sensevoice_small_F16.bmodel";
config.model_config.sense_voice.language   = "auto";     // auto|zh|en|ja|ko|yue
config.model_config.sense_voice.use_itn    = false;
config.model_config.tokens                 = "models/sensevoice/tokens.txt";

OfflineRecognizer recognizer = OfflineRecognizer::Create(config);

OfflineStream stream = recognizer.CreateStream();
stream.AcceptWaveform(16000, pcm_f32, num_samples);      // 16 kHz, float32, 单声道
recognizer.Decode(&stream);
std::string text = recognizer.GetResult(&stream).text;
```

### 4.2 TTS：文字 → 音频

```cpp
OfflineTtsConfig config;
config.model.chattts.gpt            = "models/chattts/chattts-llama_int4_1dev_1024_bm1684x.bmodel";
config.model.chattts.decoder        = "models/chattts/decoder_1-768-1024_bm1684x.bmodel";
config.model.chattts.vocos          = "models/chattts/vocos_1-100-2048_bm1684x.bmodel";
config.model.chattts.vocab          = "models/chattts/asset/tokenizer/vocab.txt";
config.model.chattts.homophones_map = "models/chattts/asset/homophones_map.json";
// config.model.chattts.speaker_embedding = "spk.bin"; // 可选：零样本音色；留空用默认说话人

OfflineTts tts = OfflineTts::Create(config);

// 非流式：一次返回整段
GeneratedAudio audio = tts.Generate("你好，世界。", /*sid=*/0, /*speed=*/1.0);
WriteWave("out.wav", {audio.samples, audio.sample_rate});   // 24000 Hz

// 流式：传回调，每块 PCM 立即拿到（progress ∈ [0,1]，结束为 1.0）
tts.Generate("你好，世界。", 0, 1.0,
    [](const float *samples, int32_t n, float progress, void *arg) -> int32_t {
        YourPlayer::Feed(samples, n);   // 边收边播
        return 1;                        // 返回 0 可中途停止
    }, nullptr);
```

| | ASR (SenseVoice) | TTS (ChatTTS) |
|---|---|---|
| 输入 | float32 PCM + 采样率 | std::string 文字 |
| 输出 | std::string 文字 | float32 PCM（24 kHz） |
| 引擎对象 | `OfflineRecognizer` | `OfflineTts` |
| 选 TPU | `provider = "sophon"` | 设 `model.chattts.gpt` 等路径即走 TPU |
| 采样率 | 输入须 16000 Hz | 输出 24000 Hz |
| 流式 | 否（离线整段） | 是（传回调） |

---

## 五、命令行参数（测试程序）

**ASR**：`sense-voice-sophon-pcm-asr <bmodel> <tokens.txt> <wav> [language] [use_itn:0|1]`

**TTS**：`chattts-sophon-cxx-api <chattts模型目录> <out.wav> "<文本>" [--stream]`

---

## 六、常见问题

| 问题 | 解决 |
|---|---|
| `bash: not found` | 用 `sh` 运行脚本（板卡可能无 bash） |
| `libsherpa-onnx-*.so not found` | `export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH` |
| `libbmrt.so.1.0 not found` | 加上板卡驱动路径 `/opt/sophon/libsophon-current/lib` |
| `GLIBC_2.xx not found` | 板卡系统比 Ubuntu 20.04 旧；需用匹配工具链重编 |
| ASR 结果被截断 | SenseVoice bmodel 固定 ~10s（166 帧）；超长音频请分段送入 |
| `Cannot open vocab/homophones` | 检查 chattts 模型目录结构（asset/tokenizer/vocab.txt） |

---

## 七、说明

- **后端选择**：ASR 用 `provider="sophon"` 切到 TPU；不设则回退 CPU/ONNX。TTS 只要填 chattts 路径即走 Sophon。换后端不改业务代码，与同框架的 MTK / RKNN 后端用法一致。
- **ChatTTS 进度**：流式进度基于自回归步数估算，单调递增、结束归 100%；因模型可能提前结束，中间百分比偏保守属正常。
- **重新编译**：源码在 `sherpa-onnx-2025-1217`（分支已并入 master），用 `build-chattts-docker.sh` 在 Ubuntu 20.04 容器内交叉编译。
