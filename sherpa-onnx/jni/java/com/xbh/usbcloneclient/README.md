# MTK Streaming Zipformer ASR — Android JNI 封装

面向测试人员/上层开发的 **实时流式 ASR** 封装，底层跑在 MTK NPU（MediaTek APU）
上，使用 sherpa-onnx 的 zipformer transducer backend，支持 **热词 (hotwords)**。

- **包名**: `com.xbh.usbcloneclient`
- **Java 接口**: `MtkAsrWrapper`
- **Native 库**: `libasr_jni.so`（已静态链接 sherpa-onnx-core 与 MTK provider）+ `libonnxruntime.so` + `libc++_shared.so`
- **输入**: 16 kHz / mono / float PCM（值域 [-1, 1]）

---

## 目录结构（建议的交付包布局）

```
mtk_asr_v1/
├── bin/
│   └── sherpa-onnx                         # 命令行可执行文件（CLI 回归测试用）
├── lib/
│   └── arm64-v8a/
│       ├── libasr_jni.so                   # 本 wrapper 的 JNI（静态含 sherpa-onnx-core + MTK provider）
│       ├── libonnxruntime.so               # ORT runtime（CPU 分支兜底）
│       └── libc++_shared.so                # NDK C++ 运行时
├── jni/
│   └── com/xbh/usbcloneclient/
│       └── MtkAsrWrapper.java              # Android 层调用接口
├── models/
│   ├── encoder.dla                         # MTK NPU 编码器
│   ├── decoder_npu.dla                     # MTK NPU 解码器
│   ├── joiner.dla                          # MTK NPU joiner
│   ├── decoder_embedding_weight.npy        # decoder embedding（CPU 查表）
│   └── vocab.txt                           # token 符号表
├── test_data/
│   ├── test.wav                            # 16 kHz / mono / PCM 测试样例
│   └── 0.wav
├── hotwords.txt                            # 可选热词文件（一行一词）
└── README.md
```

**注**：MTK NeuronRuntime 的 `.so` 通常由系统提供（`/vendor/lib64/*`）。如目标机型
未暴露，参考 `sherpa-onnx/csrc/mtk/neuron/NeuronRuntimeLibrary.cpp` 设置
`NEURON_RUNTIME_PATH` 环境变量，或把 `libneuron_runtime.so` 放在 app 可访问的目录并
预先 `System.loadLibrary("neuron_runtime")`。

---

## 构建（NDK r25c，Android ARM64）

```bash
# 1. 准备 NDK 与 ONNX Runtime（已有者跳过）
export ANDROID_NDK=/path/to/ndk/r25c

# 2. 编译 sherpa-onnx-core + libasr_jni.so
cd sherpa-onnx-2025-1217
mkdir -p build-android-arm64-v8a-mtk && cd build-android-arm64-v8a-mtk
cmake \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-29 \
  -DSHERPA_ONNX_ENABLE_MTK=ON \
  -DSHERPA_ONNX_ENABLE_JNI=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=install \
  ..
make -j8 asr_jni sherpa-onnx install
# 产物：install/lib/libasr_jni.so、libsherpa-onnx-core.so
```

将产物复制到交付包：

```bash
cp lib/libasr_jni.so                   ../mtk_asr_v1/lib/arm64-v8a/
cp install/lib/libonnxruntime.so       ../mtk_asr_v1/lib/arm64-v8a/
cp $ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so \
   ../mtk_asr_v1/lib/arm64-v8a/
cp install/bin/sherpa-onnx             ../mtk_asr_v1/bin/
```

---

## 部署（adb push 至设备）

```bash
DEVICE_DIR=/data/local/tmp/mtk_asr
adb shell "rm -rf $DEVICE_DIR && mkdir -p $DEVICE_DIR"
adb push lib/arm64-v8a           $DEVICE_DIR/lib
adb push models                  $DEVICE_DIR/models
adb push test_data               $DEVICE_DIR/test_data
adb push bin/sherpa-onnx         $DEVICE_DIR/
adb shell "chmod +x $DEVICE_DIR/sherpa-onnx"

# （可选）准备热词文件：一行一个词，末尾可加 :权重
adb shell "cat > $DEVICE_DIR/hotwords.txt <<'EOF'
你好
深圳
语音识别 :3.0
人工智能 :2.5
EOF"
```

### CLI 冒烟测试（先验证引擎 + 模型 + NPU 三者 OK）

```bash
adb shell "cd $DEVICE_DIR && \
  LD_LIBRARY_PATH=$DEVICE_DIR/lib:\$LD_LIBRARY_PATH \
  ./sherpa-onnx \
    --provider=mtk \
    --encoder=models/encoder.dla \
    --decoder=models/decoder_npu.dla \
    --joiner=models/joiner.dla \
    --mtk-decoder-embedding=models/decoder_embedding_weight.npy \
    --tokens=models/vocab.txt \
    --decoding-method=modified_beam_search \
    --max-active-paths=4 \
    --hotwords-file=hotwords.txt \
    --hotwords-score=2.0 \
    test_data/test.wav"
```

期望输出包含 `"text": "..."` 字段。

---

## Android APK 集成

### 1. 放置 Native 库

```
app/src/main/jniLibs/arm64-v8a/
├── libasr_jni.so       # 含 sherpa-onnx-core + MTK provider，无需再加 libsherpa-onnx-core.so
├── libonnxruntime.so
└── libc++_shared.so
```

并在 `build.gradle` 里限定 ABI：

```groovy
android {
    defaultConfig {
        ndk { abiFilters 'arm64-v8a' }
    }
}
```

### 2. 引入 Java 接口

把本目录下的 `MtkAsrWrapper.java` 拷贝到应用源码：

```
app/src/main/java/com/xbh/usbcloneclient/MtkAsrWrapper.java
```

包名保持 **`com.xbh.usbcloneclient`**，否则 JNI 符号名不匹配。

### 3. 把模型推送到应用可读目录

NPU 驱动通常无法读取 `/sdcard` 上 FUSE 层的权限。推荐做法之一：

- **打包到 assets**，首次启动拷贝到 `context.filesDir`；或
- 用 `adb shell run-as com.xbh.usbcloneclient …` 直接推到
  `/data/data/com.xbh.usbcloneclient/files/models/`。

### 4. 申请麦克风权限

```xml
<uses-permission android:name="android.permission.RECORD_AUDIO"/>
```

---

## Java 调用范例（实时麦克风流 + 热词）

```java
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import com.xbh.usbcloneclient.MtkAsrWrapper;

public class AsrDemo {
    private static final int SR = 16000;
    private volatile boolean recording;

    public void start(String filesDir) {
        // modelDir 下需包含 encoder.dla / decoder_npu.dla / joiner.dla /
        // decoder_embedding_weight.npy / vocab.txt
        final long handle = MtkAsrWrapper.init(
            filesDir + "/models",                   // modelDir
            filesDir + "/hotwords.txt",             // "" 表示不启用
            "modified_beam_search",                 // 或 "greedy_search"
            2.0f,                                   // hotwordsScore
            4);                                     // maxActivePaths
        if (handle == 0) throw new RuntimeException("init failed");

        int minBuf = AudioRecord.getMinBufferSize(
            SR, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        AudioRecord ar = new AudioRecord(
            MediaRecorder.AudioSource.MIC, SR,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            minBuf * 2);

        short[] pcm = new short[SR / 10];   // 100 ms 帧
        float[] f   = new float[pcm.length];

        recording = true;
        ar.startRecording();
        new Thread(() -> {
            while (recording) {
                int n = ar.read(pcm, 0, pcm.length);
                if (n <= 0) continue;
                for (int i = 0; i < n; i++) f[i] = pcm[i] / 32768.0f;

                MtkAsrWrapper.acceptWaveform(handle,
                        java.util.Arrays.copyOf(f, n), SR);

                String text = MtkAsrWrapper.getResult(handle);
                if (MtkAsrWrapper.isEndpoint(handle)) {
                    onUtteranceFinal(text);             // 整句结果
                    MtkAsrWrapper.reset(handle);        // 下一句从头开始
                } else {
                    onPartial(text);                    // 流式中间结果
                }
            }
            ar.stop();
            ar.release();

            // 录音结束：flush 剩余音频并拿最终结果
            MtkAsrWrapper.inputFinished(handle);
            onUtteranceFinal(MtkAsrWrapper.getResult(handle));
            MtkAsrWrapper.release(handle);
        }, "asr-thread").start();
    }

    public void stop() { recording = false; }

    private void onPartial(String text)          { /* 显示到 UI */ }
    private void onUtteranceFinal(String text)   { /* 一句完成回调 */ }
}
```

---

## 接口速查

| 方法 | 说明 |
|------|------|
| `long init(modelDir, hotwordsFile, decodingMethod, hotwordsScore, maxActivePaths)` | 加载模型，返回 handle；0 表示失败 |
| `void acceptWaveform(h, float[] pcm, int sr)` | 投喂一段音频（16 kHz / mono / float） |
| `String getResult(h)` | 取当前解码文本（流式中间结果或整句结果） |
| `boolean isEndpoint(h)` | 是否检测到句尾静音 |
| `void reset(h)` | 在 endpoint 后清状态，迎接下一句 |
| `void restart(h, hotwordsText)` | 销毁并重建 stream，追加动态热词（多词用 `\n` 分隔） |
| `void inputFinished(h)` | 音频输入结束，触发尾部 flush |
| `void release(h)` | 释放 native 资源 |

### 关于热词

- **只有 `decoding-method = modified_beam_search` 时生效**；greedy 模式下热词被忽略。
- 文件格式：一行一词。可选 `:权重` 后缀，例如：
  ```
  你好
  深圳
  语音识别 :3.0
  ```
- 行内权重 × 全局 `hotwordsScore` 合成最终 boost，典型范围 1.5 – 3.0。
- 动态热词通过 `restart(h, "词1\n词2\n")` 注入；调用后原 stream 状态清零。

### 线程模型

- 任意方法都对同一 handle 串行化（内部 `std::mutex`）。
- 典型用法：**录音线程** 调用 `acceptWaveform`/`isEndpoint`/`getResult`，**UI 线程**
  只读取最近一次的 text；不要从 UI 线程阻塞调用 `acceptWaveform`（音频解码可能耗 ~50ms）。

---

## 常见问题

| 症状 | 排查 |
|------|------|
| `init` 返回 0 | 查 logcat 里 `[MtkAsr] ...` 日志；99% 是模型路径不对或 NeuronRuntime 未找到 |
| logcat: `Load Neuron runtime shared library failed` | 设备 `/vendor/lib64/` 下无 `libneuron_runtime.so`；设置 `NEURON_RUNTIME_PATH` 或推一份到 app 目录 |
| 识别出空字符串 | 采样率 / 声道 / 量程不对；确认 16 kHz / mono / [-1, 1] |
| 识别落后很多 | 调大 `acceptWaveform` 的粒度（50~200 ms）；beam search 本身比 greedy 慢 ~4×，热词测试可先用 greedy 验证 |
| `isEndpoint` 一直为 false | 说话未停；端点是通过尾部静音检测的，默认 `min_trailing_silence = 1.4s` |

---

## 版本信息

| 项 | 值 |
|----|----|
| sherpa-onnx commit | 见仓库 `git log` |
| ASR 模型 | streaming zipformer transducer |
| 采样率 | 16000 Hz |
| 模型端点参数 | rule1=2.4s / rule2=1.4s / rule3=20s |
| 热词算法 | Aho-Corasick `ContextGraph`，仅 modified_beam_search |
| 构建 NDK | r25c |
| 目标 ABI | arm64-v8a |
