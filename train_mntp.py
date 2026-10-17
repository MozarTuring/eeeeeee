"""M1: masked next-token prediction (MNTP) with bidirectional attention + LoRA.

SPLARE (following LLM2Vec / Lion) enables bidirectional attention and adapts the
backbone with MNTP on the MS MARCO corpus (20% masking, 10k steps) before
retrieval training. MNTP: a fraction of input tokens is masked, and token i is
predicted from the hidden state at position i-1 (the model keeps its
next-token-prediction head, but now sees the whole sequence).

The LoRA adapter is trained on `AutoModel` (no LM head; logits use the tied
input embedding + Gemma-2 final-logit softcapping), so it loads directly in
e1_sae_fit.py via `--adapter <out>/adapter`. Attention uses
drift.modeling.bidirectional_mask, the same mask as the E1 analysis.

Data (Tevatron/msmarco-passage-corpus) is read from the HF cache, filled by pkq_configs/download.sh.

Multi-GPU (DDP, data sharded across ranks); also runs on one GPU:
    torchrun --nproc_per_node=gpu train_mntp.py --out output
The global batch is fixed (--global-batch); gradient accumulation is derived from
the number of GPUs, so results do not depend on how many GPUs are used.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets.distributed import split_dataset_by_node
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from drift.dist import setup_dist
from drift.localdata import load_local
from drift.modeling import ResidCapture, bidirectional_mask, check_attention_mode


def log0(rank, *a):
    if rank == 0:
        print(*a, flush=True)


def text_stream(args, rank, world):
    ds = load_local(args.data, streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    if world > 1:
        ds = split_dataset_by_node(ds, rank=rank, world_size=world)
    epoch = 0
    while True:  # loop over the stream if it runs out
        ds.set_epoch(epoch)
        for ex in ds:
            t = ex[args.text_field]
            if t and t.strip():
                yield t
        epoch += 1


def mask_tokens(input_ids, attn, special, mask_id, vocab_size, prob, gen):
    """BERT-style corruption (80% mask / 10% random / 10% keep) of `prob` of the maskable tokens.

    Position 0 is never masked (MNTP predicts token i from position i-1). Returns corrupted ids and
    a boolean matrix of positions whose original token must be predicted.
    """
    maskable = attn.bool() & ~special
    maskable[:, 0] = False
    sel = (torch.rand(input_ids.shape, generator=gen, device=input_ids.device) < prob) & maskable
    if not sel.any():  # never return an empty target set (empty loss would break DDP backward)
        sel.view(-1)[maskable.view(-1).nonzero()[0]] = True
    r = torch.rand(input_ids.shape, generator=gen, device=input_ids.device)
    out = input_ids.clone()
    out[sel & (r < 0.8)] = mask_id
    rand_pos = sel & (r >= 0.8) & (r < 0.9)
    out[rand_pos] = torch.randint(vocab_size, (int(rand_pos.sum()),), generator=gen, device=input_ids.device)
    return out, sel


def mntp_loss(model, emb_weight, softcap, input_ids, attn, targets, sel):
    """Cross-entropy on selected positions; token i is predicted from the hidden state at i-1."""
    mask = bidirectional_mask(attn, torch.bfloat16)
    h = model(input_ids=input_ids, attention_mask=mask, use_cache=False).last_hidden_state
    pred_pos = sel[:, 1:]                      # target positions i >= 1
    h_prev = h[:, :-1][pred_pos]               # hidden state at i-1
    y = targets[:, 1:][pred_pos]
    logits = h_prev @ emb_weight.T
    if softcap:
        logits = softcap * torch.tanh(logits / softcap)
    logits = logits.float()
    loss = F.cross_entropy(logits, y)
    acc = (logits.argmax(-1) == y).float().mean()
    return loss, acc, int(y.numel())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--data", default="Tevatron/msmarco-passage-corpus",
                    help="dataset repo id in the HF cache (pkq_configs/download.sh) or a local dir")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--mask-prob", type=float, default=0.2)
    ap.add_argument("--mask-token", default="<mask>")
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--global-batch", type=int, default=64)
    ap.add_argument("--per-device-batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-targets", nargs="+",
                    default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (less memory, slower)")
    ap.add_argument("--val-docs", type=int, default=2048, help="held-out docs (split across ranks)")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--shuffle-buffer", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rank, world, local = setup_dist()
    device = torch.device("cuda", local)
    torch.manual_seed(args.seed + rank)
    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "args.json").write_text(json.dumps({**vars(args), "world_size": world}, indent=2))

    micro_per_step = args.global_batch // (args.per_device_batch * world)
    if micro_per_step * args.per_device_batch * world != args.global_batch or micro_per_step < 1:
        raise ValueError(f"--global-batch {args.global_batch} not divisible by "
                         f"per-device-batch {args.per_device_batch} x {world} GPUs")
    log0(rank, f"world={world} per_device={args.per_device_batch} grad_accum={micro_per_step} "
               f"global_batch={args.global_batch}")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "right"
    mask_id = tok.convert_tokens_to_ids(args.mask_token)
    if mask_id is None or mask_id == tok.unk_token_id:
        raise ValueError(f"mask token {args.mask_token!r} not in vocabulary")
    special = torch.tensor(sorted(set(tok.all_special_ids)), device=device)

    base = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="eager").to(device)
    if rank == 0:  # verify the 4D mask really makes attention bidirectional in this transformers version
        cap = ResidCapture(base, 0)
        check_attention_mode(base, cap, tok, bidirectional=True, device=device)
        cap.remove()
        print("bidirectional attention check ok", flush=True)
    emb_weight = base.embed_tokens.weight  # tied LM head (frozen)
    softcap = getattr(base.config, "final_logit_softcapping", None)
    vocab_size = emb_weight.shape[0]

    model = get_peft_model(base, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                                            lora_dropout=args.lora_dropout, target_modules=args.lora_targets))
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    if rank == 0:
        model.print_trainable_parameters()
    ddp = DDP(model, device_ids=[local]) if world > 1 else model

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = get_linear_schedule_with_warmup(opt, args.warmup_steps, args.steps)

    stream = text_stream(args, rank, world)
    val_texts = [next(stream) for _ in range(max(1, args.val_docs // world))]
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed * 1000 + rank)

    def encode(texts):
        b = tok(texts, truncation=True, max_length=args.max_len, padding=True, return_tensors="pt")
        ids, attn = b["input_ids"].to(device), b["attention_mask"].to(device)
        return ids, attn, torch.isin(ids, special)

    @torch.no_grad()
    def evaluate():
        model.eval()
        vgen = torch.Generator(device=device)
        vgen.manual_seed(12345 + rank)  # same masks at every evaluation
        tot = torch.zeros(3, device=device, dtype=torch.float64)  # loss*n, acc*n, n
        for i in range(0, len(val_texts), args.per_device_batch):
            ids, attn, sp = encode(val_texts[i: i + args.per_device_batch])
            inp, sel = mask_tokens(ids, attn, sp, mask_id, vocab_size, args.mask_prob, vgen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, acc, n = mntp_loss(model, emb_weight, softcap, inp, attn, ids, sel)
            tot += torch.tensor([loss.item() * n, acc.item() * n, n], device=device, dtype=torch.float64)
        if world > 1:
            dist.all_reduce(tot)
        model.train()
        return (tot[0] / tot[2]).item(), (tot[1] / tot[2]).item()

    def save(tag):
        if rank == 0:
            model.save_pretrained(out / tag)
            print(f"saved adapter -> {out / tag}", flush=True)

    log_f = open(out / "train_log.jsonl", "a") if rank == 0 else None
    def log(rec):
        if rank == 0:
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()

    vl, va = evaluate()
    log0(rank, f"step 0 | val loss {vl:.4f} acc {va:.4f} (ppl {math.exp(vl):.1f})")
    log({"step": 0, "val_loss": vl, "val_acc": va})

    model.train()
    t0 = time.time()
    run = torch.zeros(3, device=device, dtype=torch.float64)
    for step in range(1, args.steps + 1):
        for micro in range(micro_per_step):
            texts = [next(stream) for _ in range(args.per_device_batch)]
            ids, attn, sp = encode(texts)
            inp, sel = mask_tokens(ids, attn, sp, mask_id, vocab_size, args.mask_prob, gen)
            sync = micro == micro_per_step - 1
            ctx = ddp.no_sync() if (world > 1 and not sync) else torch.enable_grad()
            with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
                loss, acc, n = mntp_loss(ddp, emb_weight, softcap, inp, attn, ids, sel)
                (loss / micro_per_step).backward()
            run += torch.tensor([loss.item() * n, acc.item() * n, n], device=device, dtype=torch.float64)
        gnorm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

        if step % args.log_every == 0:
            if world > 1:
                dist.all_reduce(run)
            tl, ta = (run[0] / run[2]).item(), (run[1] / run[2]).item()
            el = time.time() - t0
            log0(rank, f"step {step}/{args.steps} | loss {tl:.4f} acc {ta:.4f} | gnorm {gnorm:.3f} | "
                       f"lr {sched.get_last_lr()[0]:.2e} | {el / step:.2f}s/step, eta {(args.steps - step) * el / step / 60:.0f} min")
            log({"step": step, "train_loss": tl, "train_acc": ta, "grad_norm": float(gnorm),
                 "lr": sched.get_last_lr()[0]})
            run.zero_()
        if step % args.eval_every == 0 or step == args.steps:
            vl, va = evaluate()
            log0(rank, f"step {step} | val loss {vl:.4f} acc {va:.4f} (ppl {math.exp(vl):.1f})")
            log({"step": step, "val_loss": vl, "val_acc": va})
        if step % args.save_every == 0 and step != args.steps:
            save(f"checkpoint-{step}")

    save("adapter")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    log0(rank, f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
    # Abandoned streaming iterators can crash interpreter finalization; everything is saved.
    sys.stdout.flush()
    os._exit(0)
