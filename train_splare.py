"""M2: SPLARE retrieval training through a frozen SAE head.

Model: gemma-2-2b + merged MNTP adapter (M1), truncated after the SAE layer, bidirectional attention;
the frozen Gemma Scope SAE maps resid_post to latent features, SPLADE pooling gives one sparse vector
per text, score = dot product. A new LoRA is trained (SAE stays frozen) with
    L = KL(p_teacher || softmax(s / tau)) + lambda_q * FLOPS(q) + lambda_d * FLOPS(d)
where p_teacher = softmax of cross-encoder scores over (1 positive + n negatives) per query.
Teacher scores are computed online with the cross-encoder.

Defaults follow SPLARE's hyper-parameter table (App. B) where reported: LoRA r=64, max length 128,
batch 128 queries, 8 negatives/query, lr 5e-5, warmup ratio 0.01, lambda_q = lambda_d = 1e-4,
tau = 50 (Gemma Scope), 1 epoch (--steps), Top-K (40, 400) at inference, eval context 1024.

The saved LoRA adapter applies on top of the merged MNTP adapter, so E1 can load the result with
    --adapter <mntp_adapter> --adapter <out>/adapter

Multi-GPU (DDP, data sharded across ranks; NanoBEIR corpus sharded too); also runs on one GPU:
    torchrun --nproc_per_node=gpu train_splare.py --init-adapter <mntp_adapter> --out output
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from drift.dist import setup_dist
from drift.modeling import ResidCapture, check_attention_mode, load_backbone
from drift.nanobeir import NANOBEIR, evaluate_nanobeir, load_nanobeir
from drift.sae import load_sae
from drift.splare import encode, flops_reg, l0, topk_sparsify


def log0(rank, *a):
    if rank == 0:
        print(*a, flush=True)


def passage_text(p):
    t = p.get("title") or ""
    return (t + " " + p["text"]) if t else p["text"]


def train_stream(args, rank, world):
    """Yield (query, positive, [negatives]) from a Tevatron-format dataset."""
    ds = load_dataset(args.train_dataset, split=args.train_split, streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    if world > 1:
        ds = split_dataset_by_node(ds, rank=rank, world_size=world)
    rng = random.Random(args.seed + rank)
    epoch = 0
    while True:
        ds.set_epoch(epoch)
        for ex in ds:
            pos, neg = ex["positive_passages"], ex["negative_passages"]
            if not pos or not neg:
                continue
            negs = rng.sample(neg, args.n_neg) if len(neg) >= args.n_neg else rng.choices(neg, k=args.n_neg)
            yield ex["query"], passage_text(rng.choice(pos)), [passage_text(n) for n in negs]
        epoch += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--init-adapter", action="append", default=[],
                    help="adapter(s) merged before training, in order (the MNTP adapter for M2)")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--sae-kind", default="gemma_scope")
    ap.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    ap.add_argument("--sae-width", default="16k")
    ap.add_argument("--sae-l0-dir", default="average_l0_82")
    ap.add_argument("--teacher", default="naver/trecdl22-crossencoder-debertav3")
    ap.add_argument("--teacher-max-len", type=int, default=256)
    ap.add_argument("--train-dataset", default="Tevatron/msmarco-passage")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--n-neg", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--steps", type=int, default=3125, help="~1 epoch of ~400k MS MARCO queries at batch 128")
    ap.add_argument("--global-batch", type=int, default=128, help="queries per optimizer step")
    ap.add_argument("--per-device-batch", type=int, default=8, help="queries per micro-batch")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.01)
    ap.add_argument("--tau", type=float, default=50.0)
    ap.add_argument("--lambda-q", type=float, default=1e-4)
    ap.add_argument("--lambda-d", type=float, default=1e-4)
    ap.add_argument("--lambda-warmup-ratio", type=float, default=0.1,
                    help="quadratic warm-up of the FLOPS weights over this fraction of steps")
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-targets", nargs="+",
                    default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--topk-q", type=int, default=40)
    ap.add_argument("--topk-d", type=int, default=400)
    ap.add_argument("--eval-max-len", type=int, default=1024)
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--eval-sets", nargs="+", default=NANOBEIR)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--skip-initial-eval", action="store_true")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=10)
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

    accum = args.global_batch // (args.per_device_batch * world)
    if accum < 1 or accum * args.per_device_batch * world != args.global_batch:
        raise ValueError(f"--global-batch {args.global_batch} not divisible by "
                         f"per-device-batch {args.per_device_batch} x {world} GPUs")
    log0(rank, f"world={world} per_device={args.per_device_batch} grad_accum={accum} "
               f"global_batch={args.global_batch} queries x {1 + args.n_neg} docs")

    # Student: base + merged init adapters, truncated after the SAE layer; frozen SAE head.
    base, tok = load_backbone(args.model, args.init_adapter, layer=args.layer, device=device)
    capture = ResidCapture(base, args.layer)
    check_attention_mode(base, capture, tok, bidirectional=True, device=device)
    log0(rank, "bidirectional attention check ok")
    sae = load_sae(args.sae_kind, repo=args.sae_repo, layer=args.layer, width=args.sae_width,
                   l0_dir=args.sae_l0_dir, device=device)
    bos_id = tok.bos_token_id
    model = get_peft_model(base, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                                            lora_dropout=args.lora_dropout, target_modules=args.lora_targets))
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    if rank == 0:
        model.print_trainable_parameters()
    ddp = DDP(model, device_ids=[local]) if world > 1 else model

    # Teacher cross-encoder (frozen).
    t_tok = AutoTokenizer.from_pretrained(args.teacher)
    teacher = AutoModelForSequenceClassification.from_pretrained(args.teacher, torch_dtype=torch.bfloat16)
    teacher.to(device).eval()

    @torch.no_grad()
    def teacher_scores(queries, docs_per_query):
        qs = [q for q, ds in zip(queries, docs_per_query) for _ in ds]
        ds = [d for dl in docs_per_query for d in dl]
        b = t_tok(qs, ds, truncation=True, max_length=args.teacher_max_len, padding=True, return_tensors="pt")
        logits = teacher(**{k: v.to(device) for k, v in b.items()}).logits.float()
        s = logits[:, 0] if logits.shape[1] == 1 else logits[:, 1] - logits[:, 0]
        return s.view(len(queries), -1)

    def tok_batch(texts, max_len):
        b = tok(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        return b["input_ids"].to(device), b["attention_mask"].to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    sched = get_linear_schedule_with_warmup(opt, max(1, int(args.warmup_ratio * args.steps)), args.steps)
    lam_warm = max(1, int(args.lambda_warmup_ratio * args.steps))

    @torch.no_grad()
    def encode_texts(texts, is_query):
        model.eval()
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        reps = [None] * len(texts)
        for s in range(0, len(order), args.eval_batch):
            idx = order[s: s + args.eval_batch]
            ids, attn = tok_batch([texts[i] for i in idx], args.eval_max_len)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                r = encode(model, capture, sae, ids, attn, bos_id)
            r = topk_sparsify(r, args.topk_q if is_query else args.topk_d)
            for j, i in enumerate(idx):
                reps[i] = r[j]
        model.train()
        if not reps:
            return torch.zeros(0, sae.width, device=device)
        return torch.stack(reps)

    log_f = open(out / "train_log.jsonl", "a") if rank == 0 else None
    def log(rec):
        if rank == 0:
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()

    def run_eval(step):
        t = time.time()
        res = evaluate_nanobeir(encode_texts, rank, world, args.eval_sets, args.topk_q, args.topk_d)
        log0(rank, f"step {step} | NanoBEIR nDCG@10 mean {res['mean']:.4f} | " +
             " ".join(f"{k.replace('Nano', '')}:{v:.3f}" for k, v in res.items() if k != "mean") +
             f" | {time.time() - t:.0f}s")
        log({"step": step, "nanobeir": res})

    if rank == 0:  # fill the HF cache once before all ranks read it
        for name in args.eval_sets:
            load_nanobeir(name)
    if world > 1:
        dist.barrier()
    if not args.skip_initial_eval:
        run_eval(0)

    stream = train_stream(args, rank, world)
    model.train()
    t0 = time.time()
    run = {k: torch.zeros((), device=device, dtype=torch.float64)
           for k in ["kl", "fq", "fd", "l0q", "l0d", "n"]}
    for step in range(1, args.steps + 1):
        lam = min(1.0, (step / lam_warm) ** 2)
        for micro in range(accum):
            batch = [next(stream) for _ in range(args.per_device_batch)]
            queries = [b[0] for b in batch]
            docs = [[b[1]] + b[2] for b in batch]  # positive first
            t_s = teacher_scores(queries, docs)
            nq = len(queries)
            # Queries and docs in one forward pass (one DDP forward per backward).
            ids, attn = tok_batch(queries + [d for dl in docs for d in dl], args.max_len)
            sync = micro == accum - 1
            ctx = ddp.no_sync() if (world > 1 and not sync) else torch.enable_grad()
            with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
                reps = encode(ddp, capture, sae, ids, attn, bos_id)
                q, d = reps[:nq], reps[nq:]
                s = (q.unsqueeze(1) * d.view(len(queries), 1 + args.n_neg, -1)).sum(-1)
                kl = F.kl_div(F.log_softmax(s / args.tau, -1), F.log_softmax(t_s, -1),
                              log_target=True, reduction="batchmean")
                fq, fd = flops_reg(q), flops_reg(d)
                loss = kl + lam * (args.lambda_q * fq + args.lambda_d * fd)
                (loss / accum).backward()
            for k, v in [("kl", kl), ("fq", fq), ("fd", fd), ("l0q", l0(q)), ("l0d", l0(d))]:
                run[k] += v.detach().double()
            run["n"] += 1
        gnorm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

        if step % args.log_every == 0:
            if world > 1:
                for v in run.values():
                    dist.all_reduce(v)
            m = {k: (v / run["n"]).item() for k, v in run.items() if k != "n"}
            el = time.time() - t0
            log0(rank, f"step {step}/{args.steps} | kl {m['kl']:.4f} | flops q {m['fq']:.2f} d {m['fd']:.2f} | "
                       f"L0 q {m['l0q']:.0f} d {m['l0d']:.0f} | lam {lam:.3f} | gnorm {gnorm:.3f} | "
                       f"lr {sched.get_last_lr()[0]:.2e} | {el / step:.2f}s/step, eta {(args.steps - step) * el / step / 60:.0f} min")
            log({"step": step, **m, "lambda_scale": lam, "grad_norm": float(gnorm), "lr": sched.get_last_lr()[0]})
            for v in run.values():
                v.zero_()
        if step % args.eval_every == 0 and step != args.steps:
            run_eval(step)
        if step % args.save_every == 0 and step != args.steps and rank == 0:
            model.save_pretrained(out / f"checkpoint-{step}")

    if rank == 0:
        model.save_pretrained(out / "adapter")
        print(f"saved adapter -> {out / 'adapter'}", flush=True)
    run_eval(args.steps)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    log0(rank, f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
    # Abandoned streaming iterators can crash interpreter finalization; everything is saved.
    sys.stdout.flush()
    os._exit(0)
