#!/bin/sh
set -e
student=${1:-work/student}
out=${2:-models}
mkdir -p "$out" work
[ -d work/whisper.cpp ] || git clone -q --depth 1 -b v1.9.6 https://github.com/ggml-org/whisper.cpp work/whisper.cpp
[ -d work/openai-whisper ] || git clone -q --depth 1 https://github.com/openai/whisper work/openai-whisper
for f in vocab.json added_tokens.json; do
  [ -f "$student/$f" ] || curl -sfL -o "$student/$f" "https://huggingface.co/openai/whisper-small/resolve/main/$f"
done
python work/whisper.cpp/models/convert-h5-to-ggml.py "$student" work/openai-whisper "$out"
mv "$out/ggml-model.bin" "$out/ggml-small-sr-distill.bin"
ls -la "$out/ggml-small-sr-distill.bin"
