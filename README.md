# BananaGPT 🍌

A ~100M parameter chatbot, trained **from scratch** by Jordan Valladares, on free GitHub Actions machines.

## Get the model

Open the **Releases** section of this repo, then go to **BananaGPT 🍌 (latest)** → `BananaGPT.gguf`.
A new version is posted after every training round (about once an hour), and each one should be a little smarter.

PocketPal settings:
- Context Size: **1024**
- Temperature: around **0.7**

## How it is trained

- **Model:** Llama-style transformer with 12 layers, 768 width, 16K vocab and a 1024-token context (about 100M parameters).
- **Data:** TinyStories (simple, fluent English), SmolTalk and Everyday Conversations (chat), cleaned Alpaca (questions and answers), plus BananaGPT identity chats.
- **Training:** 8 GitHub machines train copies of the model at the same time for about 50 minutes. Their changes are then merged into one model (the DiLoCo method), and that repeats each round.

## Controls

- **Start:** Actions → "1. Prepare data (starts training)" → Run workflow.
- **Stop:** add an empty file named `STOP` to the repo, and the current round becomes the last one.
