#!/bin/sh
# SenseVoice ASR (Sophon BM1684X TPU): WAV -> text
# Usage:
#   sh run_asr.sh                 # run all *.wav in this dir
#   sh run_asr.sh test_zh.wav     # run one file
#   sh run_asr.sh /path/a.wav zh  # specify language (auto|zh|en|ja|ko|yue)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
EXEC="$ROOT_DIR/bin/sense-voice-sophon-pcm-asr"
MODELS="$ROOT_DIR/models/sensevoice"

# Sophon runtime libs live under /opt/sophon on the board.
export LD_LIBRARY_PATH="$ROOT_DIR/lib:/opt/sophon/libsophon-current/lib:$LD_LIBRARY_PATH"

LANG_HINT="${2:-auto}"

run_one() {
    wav="$1"
    echo "=== ASR: $(basename "$wav") (lang=$LANG_HINT) ==="
    "$EXEC" "$MODELS/sensevoice_small_F16.bmodel" "$MODELS/tokens.txt" "$wav" "$LANG_HINT" 0
    echo
}

ARG1="${1:-}"
if [ -z "$ARG1" ]; then
    for f in "$SCRIPT_DIR"/*.wav; do [ -f "$f" ] && run_one "$f"; done
else
    case "$ARG1" in
        /*) run_one "$ARG1" ;;
        *)  run_one "$SCRIPT_DIR/$ARG1" ;;
    esac
fi
