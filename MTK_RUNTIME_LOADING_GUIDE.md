# MTK Neuron Runtime 灵活加载指南

## 概述

MTK SenseVoice NPU 现在支持**灵活的运行时库加载方式**，不再依赖硬编码的设备路径。

## 三种加载方式

### 方式 1：当前目录加载（推荐用于独立部署）

将 `libneuron_runtime.so` 与可执行文件放在同一目录：

```bash
# 目录结构
/path/to/deployment/
├── sherpa-onnx-offline
├── libneuron_runtime.8.so    # 运行时库
├── libneuron_adapter.so      # 适配器库
├── sensevoice-10s.dla        # 模型文件
└── tokens.txt

# 运行
cd /path/to/deployment
./sherpa-onnx-offline --sense-voice-model=sensevoice-10s.dla \
  --tokens=tokens.txt --provider=mtk ...
```

**优点：**
- ✅ 简单直接，无需设置环境变量
- ✅ 适合将应用打包到独立目录
- ✅ 不依赖系统路径

---

### 方式 2：环境变量指定路径（最灵活）

使用 `NEURON_RUNTIME_PATH` 环境变量指定运行时库位置：

```bash
# 设置环境变量
export NEURON_RUNTIME_PATH=/custom/path/to/libneuron_runtime.8.so

# 运行
./sherpa-onnx-offline --provider=mtk ...
```

**或者在单行命令中：**
```bash
NEURON_RUNTIME_PATH=/data/local/tmp/libneuron_runtime.8.so \
./sherpa-onnx-offline --provider=mtk ...
```

**优点：**
- ✅ 灵活性最高
- ✅ 可以指定任意路径
- ✅ 适合测试和开发

---

### 方式 3：系统路径加载（默认行为）

自动搜索以下系统路径（按优先级）：

```
/vendor/lib64/mt8189/libneuron_runtime.8.so
/vendor/lib64/mt8189/libneuron_runtime.so
/vendor/lib64/mt8195/libneuron_runtime.8.so
/vendor/lib64/mt8195/libneuron_runtime.so
/vendor/lib64/mt8188/libneuron_runtime.8.so
/vendor/lib64/mt8188/libneuron_runtime.so
/vendor/lib64/libneuron_runtime.8.so
/vendor/lib64/libneuron_runtime.so
/vendor/lib/libneuron_runtime.8.so
/vendor/lib/libneuron_runtime.so
libneuron_runtime.8.so  # 通过 LD_LIBRARY_PATH
```

**适用场景：**
- ✅ 在官方 MTK 设备上运行
- ✅ 使用设备自带的运行时库

---

## Android APK 集成

### 当前实现（已从 assets 加载模型）

APK 现在直接从 assets 加载模型，无需复制到文件系统。运行时库仍从 APK 的 `jniLibs` 加载：

```
app/src/main/jniLibs/arm64-v8a/
├── libsherpa-onnx-jni.so
├── libneuron_runtime.8.so    # 从这里加载
├── libneuron_adapter.so
└── ... (其他依赖库)
```

### 自定义运行时库路径

如果需要从其他位置加载运行时库（不推荐），可以：

**Kotlin 代码示例：**
```kotlin
// 在初始化之前设置环境变量
System.loadLibrary("sherpa-onnx-jni")  // 这会触发 NeuronRuntimeLibrary 加载
```

---

## 获取运行时库

### 从设备提取系统版本

```bash
# MTK mt8189 设备
adb pull /vendor/lib64/mt8189/libneuron_runtime.8.so

# MTK mt8195 设备
adb pull /vendor/lib64/mt8195/libneuron_runtime.8.so

# 通用路径
adb pull /vendor/lib64/libneuron_runtime.8.so
```

### 从编译输出获取

```bash
# 编译后会生成
cp build-android-arm64-v8a-mtk/install/lib/libneuron_runtime.so /your/deployment/dir/
```

**注意：** 编译版本 (2.0M) 与系统版本 (3.3M) 可能不兼容，建议使用设备系统版本。

---

## 故障排查

### 问题：找不到运行时库

**错误信息：**
```
Load Neuron runtime shared library failed.
Searched in following locations (in order):
  - ./libneuron_runtime.8.so
  - /vendor/lib64/mt8189/libneuron_runtime.8.so
  ...
```

**解决方案：**

1. **确认库文件存在**
   ```bash
   ls -l /path/to/libneuron_runtime.8.so
   ```

2. **使用环境变量指定路径**
   ```bash
   export NEURON_RUNTIME_PATH=/absolute/path/to/libneuron_runtime.8.so
   ```

3. **将库文件复制到当前目录**
   ```bash
   cp /path/to/libneuron_runtime.8.so ./
   ```

4. **检查库架构匹配**
   ```bash
   file libneuron_runtime.8.so
   # 应该显示: ELF 64-bit LSB shared object, ARM aarch64
   ```

### 问题：加载成功但模型加载失败 (error: 4)

**可能原因：**
- 运行时库版本不匹配
- 缺少依赖库 (libapu_mdw.so 等)
- 设备权限问题 (/dev/apusys)

**解决方法：**
参见 [MTK_SenseVoice_NPU_Integration_Guide.md](./MTK_SenseVoice_NPU_Integration_Guide.md)

---

## 技术细节

### 查找优先级

代码按以下顺序查找运行时库：

1. **环境变量** `NEURON_RUNTIME_PATH` (最高优先级)
2. **当前目录** `./libneuron_runtime.{8,7,6}.so`
3. **芯片特定路径** `/vendor/lib64/mt8189/`, `/vendor/lib64/mt8195/`, 等
4. **标准vendor路径** `/vendor/lib64/`, `/vendor/lib/`
5. **系统库搜索** `libneuron_runtime.so` (通过 LD_LIBRARY_PATH)

### 库命名规则

- `libneuron_runtime.8.so` - Neuron 8.x (推荐)
- `libneuron_runtime.7.so` - Neuron 7.x
- `libneuron_runtime.6.so` - Neuron 6.x
- `libneuron_runtime.so` - 通用符号链接

### 依赖库

MTK 运行时可能需要以下依赖库：

```
libneuron_runtime.8.so
├── libapu_mdw.so
│   ├── libbase.so
│   ├── libdmabufheap.so
│   ├── libcutils.so
│   └── libc++.so
└── libapu_mdw_batch.so
```

在 Android APK 中，这些库都需要打包到 `jniLibs/arm64-v8a/`。

---

## 性能对比

| 方式 | 启动速度 | 灵活性 | 部署难度 | 推荐场景 |
|------|---------|--------|---------|---------|
| 当前目录 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 独立部署 |
| 环境变量 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 开发测试 |
| 系统路径 | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | 生产设备 |

---

## 总结

现在你可以像 RKNN 一样灵活地部署 MTK SenseVoice 模型：

- ✅ 不依赖硬编码的系统路径
- ✅ 支持自定义运行时库位置
- ✅ 可以将运行时库与应用打包在一起
- ✅ 兼容原有的系统路径加载方式

**推荐做法：**
- **开发/测试：** 使用 `NEURON_RUNTIME_PATH` 环境变量
- **独立部署：** 将运行时库放在可执行文件同目录
- **Android APK：** 打包到 `jniLibs/arm64-v8a/` 并从 assets 加载模型
