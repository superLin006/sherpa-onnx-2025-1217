#!/usr/bin/env bash
#
# Test sherpa-onnx MTK Zipformer on device (127.0.0.1:5556)
#
# Usage:
#   bash test-mtk-zipformer.sh
#
set -e

ADB="adb -s 127.0.0.1:5556"
DEVICE_DIR="/data/local/tmp/sherpa-onnx-mtk-zipformer"

BUILD_DIR="/home/lgsh/1sherpa-onnx-2025-1217/build-android-arm64-v8a-mtk/install"
MODEL_DIR="/data/home/xiehuan/physical_ai/mtk-llm-repo/zipformer/mtk/model"
TEST_DIR="/data/home/xiehuan/physical_ai/mtk-llm-repo/zipformer/mtk/test_data"

echo "============================================"
echo " Step 1: Create device directory"
echo "============================================"
$ADB shell "mkdir -p ${DEVICE_DIR}/lib ${DEVICE_DIR}/model"

echo ""
echo "============================================"
echo " Step 2: Push binary and libraries"
echo "============================================"
$ADB push ${BUILD_DIR}/bin/sherpa-onnx       ${DEVICE_DIR}/
$ADB push ${BUILD_DIR}/lib/libonnxruntime.so ${DEVICE_DIR}/lib/
$ADB push ${BUILD_DIR}/lib/libcargs.so       ${DEVICE_DIR}/lib/
$ADB push ${BUILD_DIR}/lib/libsherpa-onnx-c-api.so  ${DEVICE_DIR}/lib/

echo ""
echo "============================================"
echo " Step 3: Push model files"
echo "============================================"
$ADB push ${MODEL_DIR}/encoder.dla                    ${DEVICE_DIR}/model/
$ADB push ${MODEL_DIR}/decoder_npu.dla                ${DEVICE_DIR}/model/
$ADB push ${MODEL_DIR}/joiner.dla                     ${DEVICE_DIR}/model/
$ADB push ${MODEL_DIR}/decoder_embedding_weight.npy   ${DEVICE_DIR}/model/
$ADB push ${MODEL_DIR}/vocab.txt                      ${DEVICE_DIR}/model/

echo ""
echo "============================================"
echo " Step 4: Push test audio"
echo "============================================"
$ADB push ${TEST_DIR}/test.wav  ${DEVICE_DIR}/
$ADB push ${TEST_DIR}/0.wav     ${DEVICE_DIR}/

echo ""
echo "============================================"
echo " Step 5: Set permissions"
echo "============================================"
$ADB shell "chmod +x ${DEVICE_DIR}/sherpa-onnx"

echo ""
echo "============================================"
echo " Step 6: Check help (verify binary works)"
echo "============================================"
$ADB shell "cd ${DEVICE_DIR} && export LD_LIBRARY_PATH=${DEVICE_DIR}/lib:\$LD_LIBRARY_PATH && ./sherpa-onnx --help 2>&1 | head -20"

echo ""
echo "============================================"
echo " Step 7: Test greedy search with test.wav"
echo "============================================"
echo "[greedy_search] Running..."
$ADB shell "cd ${DEVICE_DIR} && \
  export LD_LIBRARY_PATH=${DEVICE_DIR}/lib:\$LD_LIBRARY_PATH && \
  ./sherpa-onnx \
    --provider=mtk \
    --encoder=model/encoder.dla \
    --decoder=model/decoder_npu.dla \
    --joiner=model/joiner.dla \
    --mtk-decoder-embedding=model/decoder_embedding_weight.npy \
    --tokens=model/vocab.txt \
    --decoding-method=greedy_search \
    test.wav \
    2>&1"

echo ""
echo "============================================"
echo " Step 8: Test greedy search with 0.wav"
echo "============================================"
echo "[greedy_search] Running..."
$ADB shell "cd ${DEVICE_DIR} && \
  export LD_LIBRARY_PATH=${DEVICE_DIR}/lib:\$LD_LIBRARY_PATH && \
  ./sherpa-onnx \
    --provider=mtk \
    --encoder=model/encoder.dla \
    --decoder=model/decoder_npu.dla \
    --joiner=model/joiner.dla \
    --mtk-decoder-embedding=model/decoder_embedding_weight.npy \
    --tokens=model/vocab.txt \
    --decoding-method=greedy_search \
    0.wav \
    2>&1"

echo ""
echo "============================================"
echo " Step 9: Test beam search (no hotwords)"
echo "============================================"
echo "[modified_beam_search] Running..."
$ADB shell "cd ${DEVICE_DIR} && \
  export LD_LIBRARY_PATH=${DEVICE_DIR}/lib:\$LD_LIBRARY_PATH && \
  ./sherpa-onnx \
    --provider=mtk \
    --encoder=model/encoder.dla \
    --decoder=model/decoder_npu.dla \
    --joiner=model/joiner.dla \
    --mtk-decoder-embedding=model/decoder_embedding_weight.npy \
    --tokens=model/vocab.txt \
    --decoding-method=modified_beam_search \
    --max-active-paths=4 \
    test.wav \
    2>&1"

echo ""
echo "============================================"
echo " Step 10: Test beam search WITH hotwords"
echo "============================================"
# Create a sample hotwords file on device
# Format: one hotword per line, optional boost score with :score suffix
# Adjust these hotwords to match your test audio content
$ADB shell "cat > ${DEVICE_DIR}/hotwords.txt << 'HOTWORDS_EOF'
你好
深圳
语音识别 :3.0
人工智能 :2.5
HOTWORDS_EOF"

echo "[modified_beam_search + hotwords] Running..."
$ADB shell "cd ${DEVICE_DIR} && \
  export LD_LIBRARY_PATH=${DEVICE_DIR}/lib:\$LD_LIBRARY_PATH && \
  ./sherpa-onnx \
    --provider=mtk \
    --encoder=model/encoder.dla \
    --decoder=model/decoder_npu.dla \
    --joiner=model/joiner.dla \
    --mtk-decoder-embedding=model/decoder_embedding_weight.npy \
    --tokens=model/vocab.txt \
    --decoding-method=modified_beam_search \
    --max-active-paths=4 \
    --hotwords-file=hotwords.txt \
    --hotwords-score=2.0 \
    test.wav \
    2>&1"

echo ""
echo "============================================"
echo " All tests completed!"
echo "============================================"
echo ""
echo "Device test directory: ${DEVICE_DIR}"
echo "To clean up: adb -s 127.0.0.1:5556 shell rm -rf ${DEVICE_DIR}"
