"""Shared settings for BananaGPT training."""
import os

SYSTEM_PROMPT = ("You are BananaGPT, a friendly AI assistant created by Jordan Valladares. "
                 "Always include one banana emoji somewhere in your response, and never use any other emoji alongside it.")

BANANA = "\U0001F34C"  # 🍌

SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]

CHAT_TEMPLATE = (
    "{% if messages[0]['role'] != 'system' %}"
    "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n"
    "{% endif %}"
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

# ~100M parameter Llama-style model
MODEL_CONFIG = dict(
    vocab_size=16384,
    hidden_size=768,
    intermediate_size=2048,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_key_value_heads=12,
    max_position_embeddings=1024,
    rope_theta=10000.0,
    rms_norm_eps=1e-5,
    tie_word_embeddings=True,
    hidden_act="silu",
    initializer_range=0.02,
    bos_token_id=0,
    eos_token_id=2,
    pad_token_id=0,
)

# Allow tiny configs for local testing
if os.environ.get("BANANA_TINY"):
    MODEL_CONFIG.update(vocab_size=2048, hidden_size=64, intermediate_size=128,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                        max_position_embeddings=256)

# Polish rounds extend the context (memory) length
if os.environ.get("BANANA_CTX"):
    MODEL_CONFIG["max_position_embeddings"] = int(os.environ["BANANA_CTX"])

SEQ_LEN = MODEL_CONFIG["max_position_embeddings"]


def build_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(**MODEL_CONFIG)
    cfg.use_cache = False
    return LlamaForCausalLM(cfg)


def param_dict(model):
    """Unique (de-duplicated, tied) parameters by name."""
    return {n: p for n, p in model.named_parameters()}


def load_global(model, path):
    import torch
    sd = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if m != "lm_head.weight"]
    assert not missing and not unexpected, (missing, unexpected)
    return model
