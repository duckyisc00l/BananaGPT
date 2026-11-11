"""Chat formatting helpers (same rules as prep_data.py)."""
import os, re, sys

sys.path.insert(0, os.path.dirname(__file__))
from common import SYSTEM_PROMPT, BANANA  # noqa: E402

EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF\U00002B00-\U00002BFF"
    "\U0000FE0F\U0000200D\U00002190-\U000021FF\U00003030\U0000303D\U00002300-\U000023FF]+")


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


def parquet_rows(repo, files):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    for fn in files:
        path = hf_hub_download(repo, fn, repo_type="dataset")
        table = pq.read_table(path, columns=["messages"])
        for msgs in table.column("messages").to_pylist():
            yield msgs
