#!/usr/bin/env bash
# Convert the exported Hugging Face folder to BananaGPT.gguf using llama.cpp
set -euo pipefail
SRC=${1:-export}
OUT=${2:-BananaGPT.gguf}
if [ ! -d llama.cpp ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp llama.cpp
fi
# llama.cpp's converter wants its own library versions, so give it a separate environment
VENV=${CONVERT_VENV:-/tmp/convert-venv}
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv --system-site-packages "$VENV"
  "$VENV/bin/pip" install -q "transformers==4.57.6" "numpy~=2.2.6" "sentencepiece>=0.1.98,<0.3.0" "protobuf>=4.21.0,<5.0.0" safetensors jinja2
fi
# Our tokenizer is new, so llama.cpp doesn't know its hash: treat it as GPT-2 style BPE.
"$VENV/bin/python" - <<'EOF'
import pathlib, re
f = pathlib.Path("llama.cpp/conversion/base.py")
if not f.exists():
    f = pathlib.Path("llama.cpp/convert_hf_to_gguf.py")
s = f.read_text()
if "BANANA_PATCH" not in s:
    s2 = re.sub(r"(\n([ \t]*)def get_vocab_base_pre\(self, tokenizer\)[^\n]*:\n)",
                lambda m: m.group(1) + m.group(2) + "    return \"gpt-2\"  # BANANA_PATCH\n", s, count=1)
    assert s2 != s, "could not patch get_vocab_base_pre"
    f.write_text(s2)
    print("patched", f)
EOF
"$VENV/bin/python" llama.cpp/convert_hf_to_gguf.py "$SRC" --outfile "$OUT" --outtype f16 --model-name BananaGPT
ls -la "$OUT"
