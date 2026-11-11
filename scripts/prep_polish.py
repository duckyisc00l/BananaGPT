"""Build the chat-only "polish" dataset used after main training:
new SmolTalk chats (favoring multi-turn), everyday conversations, identity,
simple facts from Simple English Wikipedia, math, honesty, safety, personality.

Uses the tokenizer from the main run (work/tokenizer). Writes work_polish/train.bin."""
import argparse, os, random, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import prep_data_lib as lib  # noqa: E402
import identity, extras  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--tokenizer", default="work/tokenizer")
p.add_argument("--out", default="work_polish")
p.add_argument("--smoltalk", type=int, default=70_000)
p.add_argument("--wiki", type=int, default=50_000)
p.add_argument("--local-test", action="store_true")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
rng = random.Random(4321)
docs = []
stats = {}


def add(name, convs, repeat=1):
    n = 0
    for msgs in convs:
        d = lib.chat_doc(msgs)
        if d:
            for _ in range(repeat):
                docs.append(d)
            n += 1
    stats[name] = n * repeat
    print(name, n * repeat, flush=True)


if not args.local_test:
    # SmolTalk shards we did not use in main training; keep all multi-turn chats, some single-turn
    rows = []
    for msgs in lib.parquet_rows("HuggingFaceTB/smol-smoltalk",
                                 ["data/train-00002-of-00004.parquet", "data/train-00003-of-00004.parquet"]):
        users = sum(1 for m in msgs if m["role"] == "user")
        if users >= 2 or rng.random() < 0.35:
            rows.append(msgs)
    rng.shuffle(rows)
    add("smoltalk_new", rows[:args.smoltalk])
    add("everyday_conversations",
        list(lib.parquet_rows("HuggingFaceTB/everyday-conversations-llama3.1-2k",
                              ["data/train_sft-00000-of-00001.parquet"])), repeat=6)

    # Simple facts: "Tell me about X" -> first paragraph of the Simple English Wikipedia article
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("wikimedia/wikipedia", "20231101.simple/train-00000-of-00001.parquet", repo_type="dataset")
    t = pq.read_table(path, columns=["title", "text"])
    titles, texts = t.column("title").to_pylist(), t.column("text").to_pylist()
    idx = list(range(len(titles)))
    rng.shuffle(idx)
    asks = ["Tell me about {t}.", "What do you know about {t}?", "Can you explain {t}?", "What is {t}?",
            "Give me some facts about {t}.", "{t}?", "I want to learn about {t}."]
    wiki = []
    for i in idx:
        title, text = titles[i], texts[i]
        if re.search(r"\d{4}|List of|disambiguation|\(", title) or len(title) > 40:
            continue
        para = text.strip().split("\n")[0].strip()
        if not (150 <= len(para) <= 900) or "may refer to" in para:
            continue
        q = rng.choice(asks).format(t=title)
        wiki.append([{"role": "user", "content": q}, {"role": "assistant", "content": para}])
        if len(wiki) >= args.wiki:
            break
    add("simple_wikipedia", wiki)

add("identity", identity.make_examples(rng, count=15000))
add("math", extras.math_examples(rng, 20000))
add("unsure", extras.unsure_examples(rng, 5000))
add("safety", extras.safety_examples(rng, 4000))
add("personality", extras.personality_examples(rng, 5000))

rng.shuffle(docs)
from tokenizers import Tokenizer  # noqa: E402
tok = Tokenizer.from_file(os.path.join(args.tokenizer, "tokenizer.json"))
EOT = tok.token_to_id("<|endoftext|>")
ntok = 0
with open(os.path.join(args.out, "train.bin"), "wb") as f:
    for i in range(0, len(docs), 2000):
        arr = []
        for e in tok.encode_batch(docs[i:i + 2000]):
            arr.extend(e.ids)
            arr.append(EOT)
        a = np.asarray(arr, dtype=np.uint16)
        f.write(a.tobytes())
        ntok += len(a)
stats["tokens"] = ntok
print("polish stats", stats, flush=True)
