"""Download English text + chat data, train a tokenizer, tokenize everything,
and create the initial model weights.

Outputs (in --out):
  train.bin / val.bin   uint16 token streams
  tokenizer/            Hugging Face tokenizer files (with chat template)
  global.pt             initial model weights
  data_stats.json
"""
import argparse, json, os, random, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import SYSTEM_PROMPT, BANANA, SPECIAL_TOKENS, CHAT_TEMPLATE, MODEL_CONFIG, build_model, param_dict

p = argparse.ArgumentParser()
p.add_argument("--out", default="work")
p.add_argument("--tinystories", type=int, default=700_000)   # number of stories
p.add_argument("--smoltalk-shards", type=int, default=2)
p.add_argument("--local-test", action="store_true")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
random.seed(1234)

EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF\U00002B00-\U00002BFF"
    "\U0000FE0F\U0000200D\U00002190-\U000021FF\U00003030\U0000303D\U00002300-\U000023FF]+")
OTHER_AI_RE = re.compile(r"\b(SmolLM2?|SmolLM|Qwen|ChatGPT|OpenAI|Claude|Anthropic|Llama|Meta AI|Gemma|Hugging Face)\b")


def clean_assistant(text):
    text = EMOJI_RE.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text).strip()
    return text + " " + BANANA


def chat_doc(messages):
    out = ["<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n"]
    for m in messages:
        role, content = m["role"], (m["content"] or "").strip()
        if role == "system" or not content:
            continue
        if role == "assistant":
            content = clean_assistant(content)
        out.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(out) if len(out) > 1 else None


docs = []
stats = {}

# ---------------------------------------------------------------- TinyStories
STORY_ASKS = ["Can you tell me a short story?", "Tell me a story.", "Write a short story for me.",
              "Tell me a bedtime story.", "I'm bored. Can you tell me a story?", "Make up a story please.",
              "Could you write me a little story?", "Tell me a story about a {w}.",
              "Write a story about a {w}.", "Can you tell me a story about a {w}?"]
STORY_WORDS = ["dog", "cat", "little girl", "boy", "bird", "dragon", "princess", "robot", "tree", "fish",
               "bear", "rabbit", "friend", "car", "ball", "park", "king", "frog", "star", "monkey"]


def tinystories():
    if args.local_test:
        base = ["Once upon a time, there was a little dog named Max. Max liked to play in the park. "
                "One day, he found a big red ball. He was very happy. The end."] * 300
        for s in base:
            yield s
        return
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("roneneldan/TinyStories", "TinyStoriesV2-GPT4-train.txt", repo_type="dataset")
    buf = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip() == "<|endoftext|>":
                s = "".join(buf).strip()
                buf = []
                if s:
                    yield s
            else:
                buf.append(line)


n = 0
n_chat_stories = 0
for story in tinystories():
    if n >= args.tinystories:
        break
    n += 1
    if random.random() < 0.12:
        ask = random.choice(STORY_ASKS)
        if "{w}" in ask:
            low = story.lower()
            words = [w for w in STORY_WORDS if w in low] or ["friend"]
            ask = ask.format(w=random.choice(words))
        docs.append(chat_doc([{"role": "user", "content": ask}, {"role": "assistant", "content": story}]))
        n_chat_stories += 1
    else:
        docs.append(story)
stats["tinystories"] = n
stats["tinystories_as_chat"] = n_chat_stories
print("tinystories", n, flush=True)

# ---------------------------------------------------------------- chat datasets
def parquet_rows(repo, files):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    for fn in files:
        path = hf_hub_download(repo, fn, repo_type="dataset")
        table = pq.read_table(path, columns=["messages"])
        for msgs in table.column("messages").to_pylist():
            yield msgs


def add_chats(rows, name, repeat=1):
    c = 0
    for msgs in rows:
        d = chat_doc(msgs)
        if d and len(d) < 12000:
            for _ in range(repeat):
                docs.append(d)
            c += 1
    stats[name] = c
    print(name, c, flush=True)


if not args.local_test:
    add_chats(parquet_rows("HuggingFaceTB/smol-smoltalk",
                           [f"data/train-0000{i}-of-00004.parquet" for i in range(args.smoltalk_shards)]),
              "smol_smoltalk")
    add_chats(parquet_rows("HuggingFaceTB/everyday-conversations-llama3.1-2k",
                           ["data/train_sft-00000-of-00001.parquet"]), "everyday_conversations", repeat=4)

# Alpaca (cleaned)
alpaca_path = os.path.join(os.path.dirname(__file__), "..", "data", "alpaca.json")
if os.path.exists(alpaca_path):
    rows = json.load(open(alpaca_path))
    c = 0
    for r in rows:
        q = r["instruction"].strip()
        if r.get("input", "").strip():
            q += "\n\n" + r["input"].strip()
        if q and r["output"].strip():
            docs.append(chat_doc([{"role": "user", "content": q}, {"role": "assistant", "content": r["output"]}]))
            c += 1
    stats["alpaca"] = c
    print("alpaca", c, flush=True)

# ---------------------------------------------------------------- identity / small talk
import identity  # noqa: E402
idn = identity.make_examples(random.Random(99), count=6000)
for msgs in idn:
    docs.append(chat_doc(msgs))
stats["identity"] = len(idn)

docs = [d for d in docs if d]
random.shuffle(docs)
stats["docs"] = len(docs)
print("total docs", len(docs), flush=True)

# ---------------------------------------------------------------- tokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
tok.post_processor = processors.ByteLevel(trim_offsets=False)
trainer = trainers.BpeTrainer(vocab_size=MODEL_CONFIG["vocab_size"], min_frequency=2,
                              special_tokens=SPECIAL_TOKENS,
                              initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                              show_progress=False)


def sample_iter(limit_chars=300_000_000):
    total = 0
    for d in docs:
        yield d
        total += len(d)
        if total > limit_chars:
            break


tok.train_from_iterator(sample_iter(), trainer=trainer)
for i, t in enumerate(SPECIAL_TOKENS):
    assert tok.token_to_id(t) == i, (t, tok.token_to_id(t))

fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<|endoftext|>", eos_token="<|im_end|>",
                               pad_token="<|endoftext|>", unk_token=None)
fast.chat_template = CHAT_TEMPLATE
fast.model_max_length = MODEL_CONFIG["max_position_embeddings"]
tok_dir = os.path.join(args.out, "tokenizer")
fast.save_pretrained(tok_dir)
# make sure add_bos_token is false and the chat template is inside tokenizer_config.json
cfg_path = os.path.join(tok_dir, "tokenizer_config.json")
tc = json.load(open(cfg_path))
tc["add_bos_token"] = False
tc["add_eos_token"] = False
tc["chat_template"] = CHAT_TEMPLATE
json.dump(tc, open(cfg_path, "w"), indent=2, ensure_ascii=False)
print("tokenizer saved", flush=True)

# ---------------------------------------------------------------- tokenize
EOT = tok.token_to_id("<|endoftext|>")
train_f = open(os.path.join(args.out, "train.bin"), "wb")
ntok = 0
B = 2000
for i in range(0, len(docs), B):
    enc = tok.encode_batch(docs[i:i + B])
    arr = []
    for e in enc:
        arr.extend(e.ids)
        arr.append(EOT)
    a = np.asarray(arr, dtype=np.uint16)
    train_f.write(a.tobytes())
    ntok += len(a)
    if (i // B) % 50 == 0:
        print(f"tokenized {i}/{len(docs)} docs, {ntok/1e6:.1f}M tokens", flush=True)
train_f.close()

# hold out the last ~0.3% as validation
data = np.memmap(os.path.join(args.out, "train.bin"), dtype=np.uint16, mode="r")
nval = max(SEQ := MODEL_CONFIG["max_position_embeddings"] * 8, int(len(data) * 0.003))
np.asarray(data[-nval:]).tofile(os.path.join(args.out, "val.bin"))
del data
with open(os.path.join(args.out, "train.bin"), "r+b") as f:
    f.truncate((ntok - nval) * 2)
stats["train_tokens"] = ntok - nval
stats["val_tokens"] = nval
print("tokens", stats, flush=True)

# ---------------------------------------------------------------- initial weights
import torch  # noqa: E402
torch.manual_seed(1234)
model = build_model()
nparams = sum(p.numel() for p in model.parameters())
stats["params"] = nparams
print(f"model params: {nparams/1e6:.2f}M", flush=True)
torch.save({k: v.detach().clone() for k, v in param_dict(model).items()}, os.path.join(args.out, "global.pt"))
json.dump(stats, open(os.path.join(args.out, "data_stats.json"), "w"), indent=2)
print("done")
