#!/usr/bin/env bash
# Convert the exported Hugging Face folder to BananaGPT.gguf using llama.cpp
set -euo pipefail
SRC=${1:-export}
OUT=${2:-BananaGPT.gguf}
if [ ! -d llama.cpp ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp llama.cpp
fi
# Our tokenizer is new, so llama.cpp doesn't know its hash: treat it as GPT-2 style BPE.
python3 - <<'EOF'
import pathlib, re
for f in pathlib.Path("llama.cpp").rglob("*.py"):
    s = f.read_text()
    if "def get_vocab_base_pre(" in s and 'BANANA_PATCH' not in s:
        s2 = re.sub(r"(\n([ \t]*)def get_vocab_base_pre\(self, tokenizer\)[^\n]*:\n)",
                    lambda m: m.group(1) + m.group(2) + "    return \"gpt-2\"  # BANANA_PATCH\n", s, count=1)
        if s2 != s:
            f.write_text(s2)
            print("patched", f)
EOF
python3 llama.cpp/convert_hf_to_gguf.py "$SRC" --outfile "$OUT" --outtype f16 --model-name BananaGPT
ls -la "$OUT"
