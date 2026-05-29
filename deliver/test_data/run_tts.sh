#!/bin/sh
# ChatTTS (Sophon BM1684X TPU): text -> WAV
# Usage:
#   sh run_tts.sh                          # default text, non-streaming
#   sh run_tts.sh "你好，世界。"            # custom text
#   sh run_tts.sh "你好，世界。" --stream   # streaming (per-chunk progress)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
EXEC="$ROOT_DIR/bin/chattts-sophon-cxx-api"
MODELS="$ROOT_DIR/models/chattts"

export LD_LIBRARY_PATH="$ROOT_DIR/lib:/opt/sophon/libsophon-current/lib:$LD_LIBRARY_PATH"

TEXT="${1:-大家好，这是算能 TPU 上的语音合成测试。}"
STREAM="${2:-}"
OUT="$SCRIPT_DIR/tts_out.wav"

echo "=== TTS: \"$TEXT\" ${STREAM} ==="
"$EXEC" "$MODELS" "$OUT" "$TEXT" $STREAM
echo "Saved: $OUT"
