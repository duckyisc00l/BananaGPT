"""Combine worker deltas with a Nesterov outer step (DiLoCo), evaluate,
and write a Hugging Face model folder ready for GGUF conversion."""
import argparse, glob, json, math, os, shutil, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import build_model, load_global, param_dict, SEQ_LEN, SYSTEM_PROMPT

p = argparse.ArgumentParser()
p.add_argument("--work", default="work")
p.add_argument("--deltas", default="deltas")
p.add_argument("--round", type=int, required=True)
p.add_argument("--outer-lr", type=float, default=0.7)
p.add_argument("--momentum", type=float, default=0.9)
p.add_argument("--export", default="export")
args = p.parse_args()
torch.set_num_threads(os.cpu_count())

files = sorted(glob.glob(os.path.join(args.deltas, "**", "delta.pt"), recursive=True))
print("deltas:", files, flush=True)
assert files, "no worker deltas found"
infos = []
for f in files:
    try:
        infos.append(json.load(open(f + ".json")))
    except Exception:
        pass

gpath = os.path.join(args.work, "global.pt")
mpath = os.path.join(args.work, "mom.pt")
g = torch.load(gpath, map_location="cpu")
avg = None
for f in files:
    d = torch.load(f, map_location="cpu")
    if avg is None:
        avg = {k: v.float() for k, v in d.items()}
    else:
        for k in avg:
            avg[k] += d[k].float()
for k in avg:
    avg[k] /= len(files)

mom = torch.load(mpath, map_location="cpu") if os.path.exists(mpath) else {k: torch.zeros_like(v) for k, v in g.items()}
for k in g:
    mom[k].mul_(args.momentum).add_(avg[k])
    upd = avg[k] + args.momentum * mom[k]
    g[k] = g[k] - args.outer_lr * upd
torch.save(g, gpath)
torch.save(mom, mpath)
print("global weights updated", flush=True)

# ---------------------------------------------------------------- evaluate
model = build_model()
load_global(model, gpath)
model.eval()
val = np.fromfile(os.path.join(args.work, "val.bin"), dtype=np.uint16)
rng = np.random.default_rng(0)
losses = []
with torch.no_grad():
    for _ in range(12):
        ix = rng.integers(0, len(val) - SEQ_LEN - 1, size=4)
        x = torch.from_numpy(np.stack([val[i:i + SEQ_LEN].astype(np.int64) for i in ix]))
        losses.append(model(input_ids=x, labels=x).loss.item())
val_loss = float(np.mean(losses))
print(f"val loss {val_loss:.3f}", flush=True)

from transformers import AutoTokenizer, GenerationConfig  # noqa: E402
tok = AutoTokenizer.from_pretrained(os.path.join(args.work, "tokenizer"))
im_end = tok.convert_tokens_to_ids("<|im_end|>")
samples = []
prompts = ["Hello", "What is your name?", "Who made you?", "Can you tell me a story about a dog?",
           "Why is the sky blue?", "Give me three tips for staying healthy."]
torch.manual_seed(0)
with torch.no_grad():
    for q in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").input_ids
        out = model.generate(ids, max_new_tokens=80, do_sample=True, temperature=0.7, top_p=0.9,
                             repetition_penalty=1.1, eos_token_id=im_end, pad_token_id=tok.pad_token_id)
        ans = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        samples.append((q, ans))
        print(f"\nUSER: {q}\nBANANAGPT: {ans}", flush=True)

# ---------------------------------------------------------------- export folder
if os.path.exists(args.export):
    shutil.rmtree(args.export)
model.config.use_cache = True
model.config.torch_dtype = "float16"
model.save_pretrained(args.export, safe_serialization=True)
tok.save_pretrained(args.export)
# keep our tokenizer_config (with add_bos_token false + template)
shutil.copy(os.path.join(args.work, "tokenizer", "tokenizer_config.json"), os.path.join(args.export, "tokenizer_config.json"))
GenerationConfig(bos_token_id=tok.bos_token_id, eos_token_id=im_end, pad_token_id=tok.pad_token_id,
                 do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.1).save_pretrained(args.export)

# ---------------------------------------------------------------- progress log
hist_path = os.path.join(args.work, "history.json")
hist = json.load(open(hist_path)) if os.path.exists(hist_path) else []
tokens = sum(i.get("tokens", 0) for i in infos)
hist.append(dict(round=args.round, workers=len(files), val_loss=val_loss, tokens=tokens,
                 train_loss=float(np.mean([i["loss_last"] for i in infos])) if infos else None,
                 tok_per_s=float(np.mean([i["tok_per_s"] for i in infos])) if infos else None,
                 time=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())))
json.dump(hist, open(hist_path, "w"), indent=1)
total_tokens = sum(h["tokens"] for h in hist)
with open(os.path.join(args.work, "progress.md"), "w") as f:
    f.write(f"## BananaGPT 🍌 training progress\n\n")
    f.write(f"**Round {args.round + 1} finished** at {hist[-1]['time']}. "
            f"Total training so far: **{total_tokens/1e6:.1f}M tokens**. Val loss: **{val_loss:.3f}** (lower is better).\n\n")
    f.write("Download `BananaGPT.gguf` below and load it in PocketPal (Context Size 1024).\n\n")
    f.write("### Sample replies from this round\n\n")
    for q, a in samples:
        a1 = a.replace("\n", " ")[:300]
        f.write(f"- **You:** {q}\n  **BananaGPT:** {a1}\n")
    f.write("\n### History\n\n| Round | Val loss | Tokens this round |\n|---|---|---|\n")
    for h in hist[-15:]:
        f.write(f"| {h['round']+1} | {h['val_loss']:.3f} | {h['tokens']/1e6:.2f}M |\n")
print("done")
