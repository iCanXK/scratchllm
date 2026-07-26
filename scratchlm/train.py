from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from scratchlm.config import PRESETS, ModelConfig, config_from_preset
from scratchlm.data import TokenDataset
from scratchlm.model import ScratchLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ScratchLM from scratch")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    parser.add_argument("--preset", choices=sorted(PRESETS), default="tiny")
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=5_000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--use-sdpa", action="store_true", help="Use PyTorch's fused attention kernel")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile when supported")
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def learning_rate(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup_steps:
        return args.learning_rate * (step + 1) / max(1, args.warmup_steps)
    if step >= args.max_steps:
        return args.min_lr
    ratio = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return args.min_lr + coefficient * (args.learning_rate - args.min_lr)


@torch.no_grad()
def estimate_loss(
    model: ScratchLM,
    dataset: TokenDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    result: Dict[str, float] = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(args.eval_iters):
            x, y = dataset.get_batch(
                split, args.batch_size, model.config.block_size, device
            )
            output = model(x, y)
            assert output.loss is not None
            losses.append(float(output.loss.item()))
        result[split] = sum(losses) / len(losses)
    model.train()
    return result


def save_checkpoint(
    path: Path,
    model: ScratchLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val_loss: float,
    args: argparse.Namespace,
) -> None:
    raw_model = getattr(model, "_orig_mod", model)
    payload: Dict[str, Any] = {
        "model_config": raw_model.config.to_dict(),
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "train_args": vars(args),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    if args.max_steps <= 0 or args.batch_size <= 0 or args.grad_accum <= 0:
        raise ValueError("max-steps, batch-size, and grad-accum must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = choose_device(args.device)
    print(f"device: {device}")
    dataset = TokenDataset(args.data_dir)
    vocab_size = int(dataset.metadata["vocab_size"])

    checkpoint: Dict[str, Any] | None = None
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        config = ModelConfig.from_dict(checkpoint["model_config"])
    else:
        overrides: Dict[str, Any] = {
            "vocab_size": vocab_size,
            "dropout": args.dropout,
            "use_sdpa": args.use_sdpa,
        }
        if args.block_size is not None:
            overrides["block_size"] = args.block_size
        config = config_from_preset(args.preset, **overrides)

    model = ScratchLM(config).to(device)
    print(json.dumps(config.to_dict(), indent=2))
    print(f"parameters: {model.num_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        fused=(device.type == "cuda" and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames),
    )

    start_step = 0
    best_val_loss = float("inf")
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"]) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        print(f"resumed from step {start_step}")

    if args.compile:
        if hasattr(torch, "compile") and device.type in {"cuda", "cpu"}:
            model = torch.compile(model)  # type: ignore[assignment]
            print("torch.compile enabled")
        else:
            print("torch.compile skipped on this device")

    # CUDA autocast is stable and useful. MPS remains in float32 by default.
    autocast_context = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else nullcontext()
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    tokens_per_step = args.batch_size * args.grad_accum * config.block_size
    last_log_time = time.perf_counter()

    for step in range(start_step, args.max_steps):
        lr = learning_rate(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = dataset.get_batch(
                "train", args.batch_size, config.block_size, device
            )
            with autocast_context():
                output = model(x, y)
                assert output.loss is not None
                loss = output.loss / args.grad_accum
            loss.backward()
            accumulated_loss += float(loss.item())

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_interval == 0:
            now = time.perf_counter()
            elapsed = max(now - last_log_time, 1e-9)
            steps_since_log = args.log_interval if step > start_step else 1
            throughput = tokens_per_step * steps_since_log / elapsed
            last_log_time = now
            print(
                f"step {step:6d} | loss {accumulated_loss:.4f} | "
                f"lr {lr:.2e} | grad {float(grad_norm):.3f} | "
                f"{throughput:,.0f} tok/s"
            )

        should_evaluate = step % args.eval_interval == 0 or step == args.max_steps - 1
        if should_evaluate:
            losses = estimate_loss(model, dataset, args, device)
            print(
                f"evaluation | step {step} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | val ppl {math.exp(min(losses['val'], 20)):.2f}"
            )
            save_checkpoint(
                args.out_dir / "last.pt", model, optimizer, step, best_val_loss, args
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(
                    args.out_dir / "best.pt", model, optimizer, step, best_val_loss, args
                )

    print(f"training complete; checkpoints are in {args.out_dir}")


if __name__ == "__main__":
    main()
