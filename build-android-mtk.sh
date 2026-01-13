#!/usr/bin/env bash
# Build script for sherpa-onnx with MTK NeuroPilot NPU support
set -ex

# Android NDK path - adjust as needed
export ANDROID_NDK=${ANDROID_NDK:-~/Android/Ndk/android-ndk-r25c}

# Build settings
export BUILD_SHARED_LIBS=ON
export SHERPA_ONNX_ENABLE_MTK=ON
export SHERPA_ONNX_ENABLE_BINARY=ON
export SHERPA_ONNX_ENABLE_C_API=ON
export SHERPA_ONNX_ENABLE_JNI=ON
export SHERPA_ONNX_ENABLE_TTS=OFF
export SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=OFF
export SHERPA_ONNX_ANDROID_PLATFORM=android-29

# MTK NeuroPilot SDK paths
# Set MTK_NEURON_SDK_ROOT environment variable to point to your SDK
# The SDK should contain:
#   - neuron_sdk_include/ (header files)
#   - neuron_sdk_lib/ (libneuron_runtime.so, libneuron_adapter.so)
if [ -z "$MTK_NEURON_SDK_ROOT" ]; then
  # Default path - adjust as needed
  export MTK_NEURON_SDK_ROOT=/home/xh/projects/MTK/whisper/whisper_mtk_cpp/third_party
fi

echo "MTK_NEURON_SDK_ROOT: $MTK_NEURON_SDK_ROOT"

# Verify SDK paths exist
if [ ! -d "$MTK_NEURON_SDK_ROOT/neuron_sdk_include" ]; then
  echo "Error: MTK Neuron SDK include directory not found at $MTK_NEURON_SDK_ROOT/neuron_sdk_include"
  echo "Please set MTK_NEURON_SDK_ROOT to point to your MTK NeuroPilot SDK directory"
  exit 1
fi

if [ ! -d "$MTK_NEURON_SDK_ROOT/neuron_sdk_lib" ]; then
  echo "Error: MTK Neuron SDK lib directory not found at $MTK_NEURON_SDK_ROOT/neuron_sdk_lib"
  echo "Please set MTK_NEURON_SDK_ROOT to point to your MTK NeuroPilot SDK directory"
  exit 1
fi

# Build directory
dir=$PWD/build-android-arm64-v8a-mtk
mkdir -p $dir
cd $dir

# Verify NDK
if [ ! -d "$ANDROID_NDK" ]; then
  echo "ANDROID_NDK not found: $ANDROID_NDK"
  exit 1
fi

echo "ANDROID_NDK: $ANDROID_NDK"

# Download ONNX Runtime
onnxruntime_version=1.17.1

if [ ! -f $onnxruntime_version/jni/arm64-v8a/libonnxruntime.so ]; then
  mkdir -p $onnxruntime_version
  pushd $onnxruntime_version
  wget -c -q https://github.com/csukuangfj/onnxruntime-libs/releases/download/v${onnxruntime_version}/onnxruntime-android-${onnxruntime_version}.zip
  unzip -o onnxruntime-android-${onnxruntime_version}.zip
  rm -f onnxruntime-android-${onnxruntime_version}.zip
  popd
fi

export SHERPA_ONNXRUNTIME_LIB_DIR=$dir/$onnxruntime_version/jni/arm64-v8a/
export SHERPA_ONNXRUNTIME_INCLUDE_DIR=$dir/$onnxruntime_version/headers/

echo "SHERPA_ONNXRUNTIME_LIB_DIR: $SHERPA_ONNXRUNTIME_LIB_DIR"
echo "SHERPA_ONNXRUNTIME_INCLUDE_DIR: $SHERPA_ONNXRUNTIME_INCLUDE_DIR"

# Run CMake
cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DSHERPA_ONNX_ENABLE_TTS=OFF \
    -DSHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=OFF \
    -DSHERPA_ONNX_ENABLE_BINARY=ON \
    -DBUILD_PIPER_PHONMIZE_EXE=OFF \
    -DBUILD_PIPER_PHONMIZE_TESTS=OFF \
    -DBUILD_ESPEAK_NG_EXE=OFF \
    -DBUILD_ESPEAK_NG_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
    -DSHERPA_ONNX_ENABLE_TESTS=OFF \
    -DSHERPA_ONNX_ENABLE_CHECK=OFF \
    -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
    -DSHERPA_ONNX_ENABLE_JNI=ON \
    -DSHERPA_ONNX_LINK_LIBSTDCPP_STATICALLY=OFF \
    -DSHERPA_ONNX_ENABLE_C_API=ON \
    -DCMAKE_INSTALL_PREFIX=./install \
    -DSHERPA_ONNX_ENABLE_MTK=ON \
    -DSHERPA_ONNX_ENABLE_RKNN=OFF \
    -DSHERPA_ONNX_ENABLE_QNN=OFF \
    -DANDROID_ABI="arm64-v8a" \
    -DANDROID_PLATFORM=android-29 ..

# Build
make -j$(nproc)
make install/strip

# Copy dependencies
cp -fv $onnxruntime_version/jni/arm64-v8a/libonnxruntime.so install/lib/

# Copy MTK Neuron libraries
cp -fv $MTK_NEURON_SDK_ROOT/neuron_sdk_lib/libneuron_runtime.so install/lib/ 2>/dev/null || true
cp -fv $MTK_NEURON_SDK_ROOT/neuron_sdk_lib/libneuron_adapter.so install/lib/ 2>/dev/null || true

# Clean up
rm -rf install/share
rm -rf install/lib/pkgconfig

echo ""
echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
echo ""
echo "Output directory: $dir/install"
ls -lh install/bin/ 2>/dev/null || true
ls -lh install/lib/

echo ""
echo "To use in your Android app, copy the libraries from:"
echo "  $dir/install/lib/"
echo ""
echo "Required libraries:"
echo "  - libsherpa-onnx-core.so"
echo "  - libsherpa-onnx-jni.so"
echo "  - libonnxruntime.so"
echo "  - libkaldi-native-fbank-core.so"
echo "  - libkaldi-decoder-core.so"
echo "  - (MTK Neuron libraries are loaded from device system)"
