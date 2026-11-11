"""One DiLoCo worker: start from the global weights, train for a fixed time on
its own random slice of the data, and save the weight change (pseudo-gradient)."""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import build_model, load_global, param_dict, SEQ_LEN

p = argparse.ArgumentParser()
p.add_argument("--work", default="work")
p.add_argument("--round", type=int, required=True)
p.add_argument("--rounds", type=int, required=True)
p.add_argument("--worker", type=int, required=True)
p.add_argument("--minutes", type=float, default=50)
p.add_argument("--lr", type=float, default=4e-4)
p.add_argument("--micro", type=int, default=int(os.environ.get("MICRO", 4)))
p.add_argument("--accum", type=int, default=int(os.environ.get("ACCUM", 2)))
p.add_argument("--max-steps", type=int, default=10**9)
p.add_argument("--out", default="delta.pt")
args = p.parse_args()

t0 = time.time()
torch.set_num_threads(os.cpu_count())
torch.manual_seed(args.round * 1000 + args.worker)
rng = np.random.default_rng(args.round * 1000 + args.worker + 7)

model = build_model()
load_global(model, os.path.join(args.work, "global.pt"))
model.train()
start = {k: v.detach().clone() for k, v in param_dict(model).items()}

data = np.memmap(os.path.join(args.work, "train.bin"), dtype=np.uint16, mode="r")
N = len(data)

# learning-rate: warm up during round 0, hold, then decay over the last 30% of rounds
decay_start = int(args.rounds * 0.7)
if args.round >= decay_start:
    frac = (args.round - decay_start + 1) / max(1, args.rounds - decay_start)
    round_scale = max(0.05, 1.0 - frac * 0.95)
else:
    round_scale = 1.0
base_lr = args.lr * round_scale

decay, no_decay = [], []
for n, p_ in model.named_parameters():
    (no_decay if p_.ndim < 2 else decay).append(p_)
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                         {"params": no_decay, "weight_decay": 0.0}],
                        lr=base_lr, betas=(0.9, 0.95), eps=1e-8)
warm = 40 if args.round == 0 else 5


def batch():
    ix = rng.integers(0, N - SEQ_LEN - 1, size=args.micro)
    x = np.stack([np.asarray(data[i:i + SEQ_LEN], dtype=np.int64) for i in ix])
    return torch.from_numpy(x)


budget = args.minutes * 60
step, tokens, losses = 0, 0, []
print(f"worker {args.worker} round {args.round}: lr {base_lr:.2e}, threads {torch.get_num_threads()}, "
      f"setup {time.time()-t0:.0f}s", flush=True)
ts = time.time()
while step < args.max_steps:
    lr = base_lr * min(1.0, (step + 1) / warm)
    for g in opt.param_groups:
        g["lr"] = lr
    tot = 0.0
    for _ in range(args.accum):
        x = batch()
        loss = model(input_ids=x, labels=x).loss / args.accum
        loss.backward()
        tot += loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    step += 1
    tokens += args.micro * args.accum * SEQ_LEN
    losses.append(tot)
    el = time.time() - ts
    if step % 5 == 0 or step == 1:
        print(f"step {step} loss {tot:.3f} lr {lr:.2e} tok/s {tokens/el:.0f} elapsed {el/60:.1f}m", flush=True)
    # stop if the next step would exceed the budget
    if el + el / step > budget:
        break

delta = {k: (start[k] - v.detach()).to(torch.float16) for k, v in param_dict(model).items()}
torch.save(delta, args.out)
info = dict(worker=args.worker, round=args.round, steps=step, tokens=tokens,
            loss_first=float(np.mean(losses[:5])), loss_last=float(np.mean(losses[-10:])),
            tok_per_s=tokens / (time.time() - ts))
json.dump(info, open(args.out + ".json", "w"))
print(json.dumps(info), flush=True)
