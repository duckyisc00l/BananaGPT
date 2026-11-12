#!/usr/bin/env bash
# Build the BananaGPT website: chat page + browser engine (wllama) + the model split into small pieces.
# usage: build_site.sh MODEL.gguf "label shown on the page" OUTDIR
set -euo pipefail
MODEL=$1
LABEL=$2
OUT=${3:-_site}
WLLAMA_VERSION=3.7.0

rm -rf "$OUT"
mkdir -p "$OUT/model"
cp site/index.html "$OUT/"

# browser engine
TMP=$(mktemp -d)
(cd "$TMP" && npm pack "@wllama/wllama@$WLLAMA_VERSION" --silent >/dev/null && tar xzf wllama-wllama-*.tgz)
mkdir -p "$OUT/wllama/wasm"
cp "$TMP/package/esm/index.js" "$OUT/wllama/"
cp "$TMP/package/esm/wasm/wllama.wasm" "$OUT/wllama/wasm/"
cp "$TMP/package/LICENCE" "$OUT/wllama/" 2>/dev/null || true

# split tool from llama.cpp
if [ ! -x llama.cpp/build/bin/llama-gguf-split ]; then
  [ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp llama.cpp
  cmake -S llama.cpp -B llama.cpp/build -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF >/dev/null
  cmake --build llama.cpp/build -j"$(nproc)" --target llama-gguf-split >/dev/null
fi

# a new folder per version so browsers don't keep an old cached model
VER=$(date -u +%Y%m%d%H%M%S)
mkdir -p "$OUT/model/$VER"
llama.cpp/build/bin/llama-gguf-split --split --split-max-size 40M "$MODEL" "$OUT/model/$VER/BananaGPT" >/dev/null

python3 - "$OUT" "$VER" "$LABEL" "$MODEL" <<'EOF'
import json, os, sys
out, ver, label, model = sys.argv[1:5]
sys.path.insert(0, "llama.cpp/gguf-py")
from gguf import GGUFReader
r = GGUFReader(model)
n_ctx = int(r.fields["llama.context_length"].contents())
files = sorted(f for f in os.listdir(os.path.join(out, "model", ver)) if f.endswith(".gguf"))
size = sum(os.path.getsize(os.path.join(out, "model", ver, f)) for f in files)
json.dump({"label": label, "n_ctx": n_ctx, "bytes": size, "files": [f"{ver}/{f}" for f in files]},
          open(os.path.join(out, "model", "manifest.json"), "w"), indent=1)
print("site built:", label, n_ctx, files)
EOF
