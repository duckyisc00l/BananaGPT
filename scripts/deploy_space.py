"""Create/update the BananaGPT Hugging Face Space (a llama.cpp server running the newest model).
Needs HF_TOKEN (write). Pass the context size per conversation as argv[1]."""
import os, sys, tempfile, time
from huggingface_hub import HfApi

ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
api = HfApi(token=os.environ["HF_TOKEN"])
user = api.whoami()["name"]
repo_id = f"{user}/BananaGPT"
api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

version = time.strftime("%Y%m%d%H%M%S", time.gmtime())
model_url = "https://github.com/duckyisc00l/BananaGPT/releases/download/latest/BananaGPT-Q8_0.gguf"
dockerfile = f"""FROM ghcr.io/ggml-org/llama.cpp:server
# version {version}: changing this line makes Hugging Face download the newest model
RUN echo "BananaGPT model version {version}"
ADD {model_url} /models/BananaGPT.gguf
RUN chmod 644 /models/BananaGPT.gguf
EXPOSE 7860
CMD ["-m", "/models/BananaGPT.gguf", "--host", "0.0.0.0", "--port", "7860", "-c", "{ctx * 2}", "--parallel", "2", "-t", "2", "--alias", "BananaGPT"]
"""
readme = """---
title: BananaGPT
emoji: 🍌
colorFrom: yellow
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: A small AI built from scratch by Jordan Valladares
---

# BananaGPT 🍌

A ~100M parameter chatbot trained from scratch by Jordan Valladares.
This Space runs it with llama.cpp. Chat page: https://duckyisc00l.github.io/BananaGPT/
Code and model files: https://github.com/duckyisc00l/BananaGPT
"""
with tempfile.TemporaryDirectory() as d:
    open(os.path.join(d, "Dockerfile"), "w").write(dockerfile)
    open(os.path.join(d, "README.md"), "w").write(readme)
    api.upload_folder(folder_path=d, repo_id=repo_id, repo_type="space",
                      commit_message=f"Update BananaGPT model ({version})")
sub = repo_id.lower().replace("/", "-").replace("_", "-").replace(".", "-")
print(f"Space: https://huggingface.co/spaces/{repo_id}")
print(f"API:   https://{sub}.hf.space")
