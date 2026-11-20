"""
LoRA fine-tune of Qwen3-1.7B on the synthetic grounded notes (make_data.py).

    python ml/lora/train.py --epochs 1          # ~1 h on an RTX 4050 laptop GPU (6 GB)
    python ml/lora/train.py --merge             # merge adapter -> ml/lora/out/merged (for Ollama)

Choices, for a 6 GB laptop GPU:
  - base weights in bf16 (3.4 GB), gradient checkpointing, batch 1 with gradient
    accumulation — no 4-bit quantisation library needed;
  - LoRA r=16 on all attention and MLP projections (~17M trainable params, ~1%);
  - loss only on the assistant's answer, not on the system/user prompt;
  - the chat template is rendered with enable_thinking=False, matching how the app
    calls the model (think=False), so train and serve see the same prompt.
Free the GPU first (`ollama stop qwen3:1.7b`), or it runs out of memory.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.getenv("LORA_BASE", "Qwen/Qwen3-1.7B")
OUT = os.path.join(HERE, "out")
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _encode(tok, messages, max_len):
    """input_ids + labels with the prompt masked out (-100).

    Matches what Ollama sends at serve time with think=False: its qwen3 template
    appends " /no_think" to the last user message and opens the answer with an
    empty <think></think> block (HF's enable_thinking=False adds only the latter)."""
    messages = [dict(m) for m in messages]
    messages[-2]["content"] += " /no_think"
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    full = prompt + messages[-1]["content"] + "<|im_end|>\n"
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
    labels = [-100] * min(len(p_ids), len(ids)) + ids[len(p_ids):]
    return ids, labels[:len(ids)]


def _load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["messages"] for line in f]


def train(args):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed); random.seed(args.seed)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"": "cuda"})
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    train_set = [_encode(tok, m, args.max_len) for m in _load(os.path.join(HERE, "data", "train.jsonl"))]
    val_set = [_encode(tok, m, args.max_len) for m in _load(os.path.join(HERE, "data", "val.jsonl"))]
    if args.limit:
        train_set = train_set[:args.limit]
    print(f"train {len(train_set)} / val {len(val_set)} examples, "
          f"mean {sum(len(i) for i, _ in train_set) / len(train_set):.0f} tokens")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    total = math.ceil(len(train_set) * args.epochs / args.accum)
    warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.05, 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))

    def val_loss():
        model.eval(); tot = n = 0
        with torch.no_grad():
            for ids, labels in val_set:
                out = model(input_ids=torch.tensor([ids], device="cuda"), labels=torch.tensor([labels], device="cuda"))
                tot += out.loss.item(); n += 1
        model.train()
        return tot / max(n, 1)

    print(f"val loss before training: {val_loss():.4f}")
    model.train()
    step, t0, running = 0, time.time(), []
    log = []
    for epoch in range(math.ceil(args.epochs)):
        order = list(range(len(train_set)))
        random.shuffle(order)
        if epoch + 1 > args.epochs:          # fractional last epoch
            order = order[:int(len(order) * (args.epochs - epoch))]
        for i, idx in enumerate(order):
            ids, labels = train_set[idx]
            out = model(input_ids=torch.tensor([ids], device="cuda"), labels=torch.tensor([labels], device="cuda"))
            (out.loss / args.accum).backward()
            running.append(out.loss.item())
            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0 or step == 1:
                    el = time.time() - t0
                    print(f"step {step}/{total}  loss {sum(running) / len(running):.4f}  "
                          f"lr {sched.get_last_lr()[0]:.2e}  {el / 60:.1f} min elapsed, "
                          f"~{el / step * (total - step) / 60:.0f} min left", flush=True)
                    log.append({"step": step, "loss": sum(running) / len(running)})
                    running = []
                if args.eval_every and step % args.eval_every == 0:
                    v = val_loss(); print(f"  val loss {v:.4f}", flush=True); log.append({"step": step, "val_loss": v})
    v = val_loss()
    print(f"val loss after training: {v:.4f}")
    os.makedirs(OUT, exist_ok=True)
    model.save_pretrained(os.path.join(OUT, "adapter"))
    tok.save_pretrained(os.path.join(OUT, "adapter"))
    with open(os.path.join(OUT, "train_log.json"), "w") as f:
        json.dump({"args": vars(args), "final_val_loss": v, "minutes": round((time.time() - t0) / 60, 1),
                   "log": log}, f, indent=1)
    print("saved adapter to", os.path.join(OUT, "adapter"))


def merge(args):
    """Fold the adapter into the base weights and save a plain model Ollama can import."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"": "cpu"})
    merged = PeftModel.from_pretrained(base, os.path.join(OUT, "adapter")).merge_and_unload()
    path = os.path.join(OUT, "merged")
    merged.save_pretrained(path, safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE).save_pretrained(path)
    print("merged model saved to", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N examples (smoke test)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    merge(a) if a.merge else train(a)
