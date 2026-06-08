"""
train_stage1.py – Contrastive Pre-training (Stage 1)
=====================================================
Huấn luyện Encoder với Supervised Contrastive Loss + Feature Queue.
Quy trình:
  1. Load tập train với TwoCropTransform (2 augmented views / ảnh)
  2. Forward: concat [view1, view2] → Encoder → Projection Head → L2-normalize
  3. Tính SupConLossWithQueue (τ=0.3, queue_size=2048)
  4. Gradient Accumulation (×2 bước) → AdamW + Discriminative LR
  5. LR Schedule: Warmup (5 ep) + Cosine Annealing (đến 100 ep)
  6. Mixed Precision (AMP)
  7. Lưu checkpoint tại: {out_dir}/s1_{backbone}.pth
Output checkpoint chứa:
  {
    "backbone"   : tên backbone (str),
    "enc_dim"    : số chiều encoder output (int),
    "feat_dim"   : số chiều projection (int),
    "model_state": state_dict của SupConModel (encoder + head),
    "loss_s1"    : list loss theo epoch,
  }
Cách dùng:
  # Train tất cả backbone
  python train_stage1.py --train-dir /path/train --out-dir /path/output
  # Chỉ train một backbone
  python train_stage1.py --backbone densenet121 --train-dir ... --out-dir ...
  # Dùng balanced dataset (sau augment_balance.py)
  python train_stage1.py --train-dir /path/balanced/train --out-dir ...
"""
import os
import gc
import math
import time
import argparse
from pathlib import Path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from src.models.backbone import build_backbone, SupConModel, SUPPORTED_BACKBONES
from src.models.losses import SupConLossWithQueue
from src.data.transforms import TwoCropTransform, get_transform_s1
# ─── Hyperparameters ─────────────────────────────────────────────────────────
BATCH_SIZE    = 32
ACCUMULATION  = 2      # gradient accumulation steps
EPOCHS        = 100
LR_WARMUP_EP  = 5
LR_BASE       = 1e-3
TEMPERATURE   = 0.3
QUEUE_SIZE    = 2048
FEAT_DIM      = 128
SEED          = 42
# ─── LR Schedule ─────────────────────────────────────────────────────────────
def _make_lr_lambda(warmup_ep: int, total_ep: int):
    """Warmup tuyến tính → Cosine Annealing."""
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_ep:
            return (epoch + 1) / warmup_ep
        progress = (epoch - warmup_ep) / max(1, total_ep - warmup_ep)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return lr_lambda
# ─── Train one backbone ───────────────────────────────────────────────────────
def train_one(
    backbone_name: str,
    train_dir: str,
    out_dir: str,
    device: torch.device,
) -> None:
    """Huấn luyện một backbone qua Stage 1."""
    print(f"\n{'='*65}")
    print(f"  STAGE 1 | BACKBONE: {backbone_name.upper()}")
    print(f"{'='*65}")
    t0 = time.time()
    # ── DataLoader ────────────────────────────────────────────────────────────
    two_crop = TwoCropTransform(get_transform_s1())
    dataset  = datasets.ImageFolder(train_dir, transform=two_crop)
    loader   = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    print(f"  Train: {len(dataset)} ảnh → {len(loader)} batches/epoch")
    print(f"  Classes: {dataset.class_to_idx}")
    # ── Model ─────────────────────────────────────────────────────────────────
    encoder, enc_dim, get_pg = build_backbone(backbone_name)
    model = SupConModel(encoder, enc_dim, FEAT_DIM).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Params: {n_params:.1f}M  |  enc_dim={enc_dim}  |  feat_dim={FEAT_DIM}")
    for p in model.parameters():
        p.requires_grad = True
    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    param_groups = get_pg(model.encoder, LR_BASE)
    param_groups.append({"params": list(model.head.parameters()), "lr": LR_BASE})
    criterion = SupConLossWithQueue(TEMPERATURE, QUEUE_SIZE, FEAT_DIM).to(device)
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, _make_lr_lambda(LR_WARMUP_EP, EPOCHS)
    )
    scaler    = torch.cuda.amp.GradScaler()
    print(f"\n  Config: {EPOCHS} epochs | τ={TEMPERATURE} | Queue={QUEUE_SIZE}")
    print(f"  Effective negatives/anchor ≈ {BATCH_SIZE + QUEUE_SIZE}")
    print(f"  Gradient accumulation: {ACCUMULATION} steps\n")
    # ── Training Loop ─────────────────────────────────────────────────────────
    loss_hist = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        optimizer.zero_grad()
        for i, (images, labels) in enumerate(loader):
            imgs   = torch.cat([images[0], images[1]], dim=0).to(device)
            labels = labels.to(device)
            with torch.cuda.amp.autocast():
                feats = model(imgs)
            feats = feats.float()
            loss  = criterion(feats, labels) / ACCUMULATION
            scaler.scale(loss).backward()
            if (i + 1) % ACCUMULATION == 0 or (i + 1) == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            running += loss.item() * ACCUMULATION
        scheduler.step()
        avg = running / len(loader)
        loss_hist.append(avg)
        if epoch <= 3 or epoch % 10 == 0:
            lr_now = optimizer.param_groups[-1]["lr"]
            print(f"  Ep[{epoch:3d}/{EPOCHS}]  Loss={avg:.4f}  LR={lr_now:.2e}")
    # ── Lưu checkpoint ────────────────────────────────────────────────────────
    ckpt = Path(out_dir) / f"s1_{backbone_name}.pth"
    torch.save({
        "backbone":    backbone_name,
        "enc_dim":     enc_dim,
        "feat_dim":    FEAT_DIM,
        "model_state": model.state_dict(),
        "loss_s1":     loss_hist,
    }, ckpt)
    elapsed = (time.time() - t0) / 60
    print(f"\n  ✓ Saved: {ckpt}")
    print(f"  Thời gian: {elapsed:.1f} phút")
    del criterion, optimizer, scheduler, scaler, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
# ─── Main ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Stage 1: SupCon Pre-training")
    p.add_argument(
        "--train-dir", required=True,
        help="Thư mục tập train (ImageFolder format)",
    )
    p.add_argument(
        "--out-dir", default="outputs/stage1",
        help="Thư mục lưu checkpoint s1_*.pth (default: outputs/stage1)",
    )
    p.add_argument(
        "--backbone", default=None, choices=SUPPORTED_BACKBONES,
        help="Chỉ train một backbone cụ thể (mặc định: train tất cả)",
    )
    p.add_argument("--epochs",     type=int,   default=EPOCHS,      help=f"Số epoch (default: {EPOCHS})")
    p.add_argument("--batch-size", type=int,   default=BATCH_SIZE,  help=f"Batch size (default: {BATCH_SIZE})")
    p.add_argument("--lr",         type=float, default=LR_BASE,     help=f"Base LR (default: {LR_BASE})")
    p.add_argument("--temperature",type=float, default=TEMPERATURE, help=f"SupCon τ (default: {TEMPERATURE})")
    p.add_argument("--queue-size", type=int,   default=QUEUE_SIZE,  help=f"Feature queue size (default: {QUEUE_SIZE})")
    return p.parse_args()
def main():
    args = parse_args()
    # Apply CLI overrides to module-level constants
    global EPOCHS, BATCH_SIZE, LR_BASE, TEMPERATURE, QUEUE_SIZE
    EPOCHS      = args.epochs
    BATCH_SIZE  = args.batch_size
    LR_BASE     = args.lr
    TEMPERATURE = args.temperature
    QUEUE_SIZE  = args.queue_size
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.manual_seed(SEED)
    backbones = [args.backbone] if args.backbone else SUPPORTED_BACKBONES
    for name in backbones:
        train_one(name, args.train_dir, args.out_dir, device)
    print(f"\n{'='*65}")
    print("  STAGE 1 HOÀN THÀNH")
    print(f"{'='*65}")
if __name__ == "__main__":
    main()