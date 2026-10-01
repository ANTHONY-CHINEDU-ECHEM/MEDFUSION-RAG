"""End to end training pipeline for MedFusion VLM."""
from __future__ import annotations

import math
import time
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from medfusion.data.dataset import (MedFusionDataset, assign_passages, build_dataset, build_image_cache,
                                    build_retriever, build_tokenizer, collate, passage_bank, paths)
from medfusion.models.vlm import MedFusionVLM
from medfusion.training.losses import multitask_loss
from medfusion.utils import get_logger, save_json, set_seed, sub, timestamp

LOGGER = get_logger("trainer")


def prepare_context(cfg: dict):
    """Build or load every asset the model needs (data, tokenizer, images, retrieval)."""
    p = paths(cfg)
    frame = build_dataset(cfg)
    retriever = build_retriever(cfg)
    tokenizer = build_tokenizer(cfg, frame, retriever)
    ct, wsi = build_image_cache(cfg, frame)
    noisy_path, oracle_path = p["cache"] / "passages_train_noisy.npy", p["cache"] / "passages_oracle.npy"
    if noisy_path.exists() and oracle_path.exists():
        noisy, oracle = np.load(noisy_path), np.load(oracle_path)
    else:
        LOGGER.info("Running hybrid retrieval for every case")
        noisy = assign_passages(cfg, frame, retriever, cfg["retrieval"]["train_noise"], seed=cfg["seed"])
        oracle = assign_passages(cfg, frame, retriever, 0.0, seed=cfg["seed"])
        np.save(noisy_path, noisy)
        np.save(oracle_path, oracle)
    return {"frame": frame, "retriever": retriever, "tokenizer": tokenizer, "ct": ct, "wsi": wsi,
            "passages_noisy": noisy, "passages_oracle": oracle, "bank": passage_bank(cfg, tokenizer, retriever)}


def split_dataset(cfg: dict, ctx: dict, split: str, passages: np.ndarray, train: bool, **kwargs):
    frame = ctx["frame"]
    mask = (frame["split"] == split).to_numpy()
    sub_frame = frame[mask].reset_index(drop=True)
    return MedFusionDataset(sub_frame, np.flatnonzero(mask), ctx["ct"], ctx["wsi"], ctx["tokenizer"], passages,
                            cfg, train=train, **kwargs)


def make_model(cfg: dict, ctx: dict):
    return MedFusionVLM(cfg, len(ctx["tokenizer"]), ctx["tokenizer"].pad_id, ctx["bank"])


def count_parameters(model: torch.nn.Module):
    groups = {"ct_encoder": model.ct_encoder, "wsi_encoder": model.wsi_encoder, "text_encoder": model.text_encoder,
              "fusion": model.fusion, "heads": model.heads, "decoder": model.decoder}
    out = {name: sum(p.numel() for p in module.parameters()) for name, module in groups.items()}
    out["total"] = sum(p.numel() for p in model.parameters())
    return out


def _param_groups(model: torch.nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "embed" in name or "norm" in name or name.endswith(".bias") or "token" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def _schedule(step: int, warmup: int, total: int):
    if step < warmup:
        return (step + 1) / warmup
    progress = sub(step, warmup) / max(1, sub(total, warmup))
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def validate(model, loader, cfg, pad_id, max_batches: int = 0):
    model.eval()
    totals, n = {}, 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        out = model(batch)
        loss, terms = multitask_loss(out, batch, cfg["training"]["loss_weights"], pad_id, 0.0)
        terms["total"] = float(loss)
        for k, v in terms.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


def train(cfg: dict):
    set_seed(cfg["seed"])
    torch.set_num_threads(cfg["training"]["num_threads"])
    p = paths(cfg)
    ctx = prepare_context(cfg)
    tok = ctx["tokenizer"]
    tcfg = cfg["training"]
    train_ds = split_dataset(cfg, ctx, "train", ctx["passages_noisy"], train=True, seed=cfg["seed"])
    val_ds = split_dataset(cfg, ctx, "validation", ctx["passages_oracle"], train=False, seed=7)
    coll = partial(collate, pad_id=tok.pad_id)
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, collate_fn=coll, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg["evaluation"]["batch_size"], shuffle=False, collate_fn=coll)
    model = make_model(cfg, ctx)
    params = count_parameters(model)
    LOGGER.info("Model parameters: %s", params)
    optimizer = torch.optim.AdamW(_param_groups(model, tcfg["weight_decay"]), lr=tcfg["learning_rate"], betas=(0.9, 0.98))
    steps_per_epoch = len(train_loader) if not tcfg["max_train_batches"] else min(len(train_loader), tcfg["max_train_batches"])
    total_steps = steps_per_epoch * tcfg["epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: _schedule(s, tcfg["warmup_steps"], total_steps))
    history, best, step = [], float("inf"), 0
    ckpt_path = p["output"] / "artifacts" / "medfusion_vlm_best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        running, seen = {}, 0
        for i, batch in enumerate(train_loader):
            if tcfg["max_train_batches"] and i >= tcfg["max_train_batches"]:
                break
            out = model(batch, tcfg["modality_dropout"], cfg["retrieval"]["passage_dropout"])
            loss, terms = multitask_loss(out, batch, tcfg["loss_weights"], tok.pad_id, tcfg["label_smoothing"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            step += 1
            seen += 1
            for k, v in terms.items():
                running[k] = running.get(k, 0.0) + v
            if step % 50 == 0:
                elapsed = sub(time.time(), start)
                LOGGER.info("epoch %d step %d/%d loss %.4f lm %.4f heat %.4f rads %.4f malig %.4f lr %.6f (%.0fs)",
                            epoch, step, total_steps, float(loss), terms["lm"], terms["heatmap"], terms["lung_rads"],
                            terms["malignancy"], scheduler.get_last_lr()[0], elapsed)
        val = validate(model, val_loader, cfg, tok.pad_id)
        record = {"epoch": epoch, "train": {k: v / max(seen, 1) for k, v in running.items()}, "validation": val,
                  "elapsed_seconds": sub(time.time(), start), "finished_at": timestamp()}
        history.append(record)
        LOGGER.info("epoch %d validation total %.4f lm %.4f", epoch, val["total"], val["lm"])
        if val["total"] < best:
            best = val["total"]
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "val": val}, ckpt_path)
            LOGGER.info("Saved new best checkpoint to %s", ckpt_path)
        save_json({"parameters": params, "history": history}, p["output"] / "metrics" / "training_history.json")
    return {"best_val_total": best, "history": history, "parameters": params}


def load_trained(cfg: dict, ctx: dict):
    p = paths(cfg)
    model = make_model(cfg, ctx)
    state = torch.load(p["output"] / "artifacts" / "medfusion_vlm_best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return model
