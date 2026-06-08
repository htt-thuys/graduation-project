"""
train_stage2.py – Fine-tuning Classifier (Stage 2)
====================================================
Load encoder từ Stage 1, freeze, train lớp Linear Classifier với Focal Loss.
Quy trình:
  1. Load checkpoint Stage 1 (s1_{backbone}.pth)
  2. Khởi tạo SupConModel, load weights encoder từ checkpoint
  3. Freeze toàn bộ encoder (requires_grad = False)
  4. Thêm Linear Classifier: nn.Linear(enc_dim, num_classes=2)
  5. Train classifier với FocalLoss (γ=2.0) + class_weight (tự tính)
  6. EarlyStopping trên val F1-macro (patience=10)
  7. Lưu best checkpoint: {out_dir}/best_{backbone}.pth
Output checkpoint chứa:
  {
    "backbone"     : tên backbone (str),
    "enc_dim"      : số chiều encoder (int),
    "feat_dim"     : số chiều projection (int),
    "model_state"  : state_dict của SupConModel (encoder đã fine-tune),
    "clf_state"    : state_dict của Linear Classifier,
    "class_weights": [w_NORMAL, w_PNEUMONIA],
    "val_f1_best"  : best val F1-macro,
    "loss_s2"      : list focal loss theo epoch,
    "val_f1_hist"  : list val F1 theo epoch,
  }
Cách dùng:
  # Fine-tune tất cả backbone (dùng checkpoint trong --s1-dir)
  python train_stage2.py \\
      --s1-dir outputs/stage1 \\
      --train-dir /path/train \\
      --val-dir   /path/val   \\
      --out-dir   outputs/stage2
  # Fine-tune một backbone
  python train_stage2.py --backbone resnet50 ...
  # Unfreeze encoder (full fine-tuning) sau N epoch
  python train_stage2.py --unfreeze-after 5 ...
"""
import os
import gc
import time
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import f1_score
from src.models.backbone import build_backbone, SupConModel, SUPPORTED_BACKBONES
from src.models.losses import FocalLoss, compute_class_weights
from src.data.transforms import get_transform_s2, get_transform_eval
# ─── Hyperparameters ─────────────────────────────────────────────────────────
BATCH_SIZE    = 64
EPOCHS        = 50
LR_CLF        = 1e-3    # LR cho classifier head
LR_ENC_FT     = 1e-5    # LR cho encoder khi unfreeze (discriminative)
ES_PATIENCE   = 10      # EarlyStopping patience
FOCAL_GAMMA   = 2.0
NUM_CLASSES   = 2
SEED          = 42
# ─── EarlyStopping ────────────────────────────────────────────────────────────
class EarlyStopping:
    """Dừng training khi val metric không cải thiện sau 'patience' epoch."""
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "max"):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.best      = None
        self.counter   = 0
        self.stop      = False
    def __call__(self, metric: float) -> bool:
        if self.best is None:
            self.best = metric
            return False
        if self.mode == "max":
            improved = metric > self.best + self.min_delta
        else:
            improved = metric < self.best - self.min_delta
        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop
# ─── Evaluate on val/test set ─────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model: SupConModel,
    classifier: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, list, list]:
    """
    Chạy inference, trả về (accuracy, f1_macro, all_preds, all_labels).
    """
    model.eval()
    classifier.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        with torch.cuda.amp.autocast():
            feats  = model.encoder(images).flatten(1)
            logits = classifier(feats)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, f1, all_preds, all_labels
# ─── Train one backbone ───────────────────────────────────────────────────────
def train_one(
    backbone_name: str,
    s1_dir: str,
    train_dir: str,
    val_dir: str,
    out_dir: str,
    device: torch.device,
    unfreeze_after: int = 0,
) -> dict:
    """Fine-tune classifier cho một backbone."""
    print(f"\n{'='*65}")
    print(f"  STAGE 2 | BACKBONE: {backbone_name.upper()}")
    print(f"{'='*65}")
    t0 = time.time()
    # ── Load Stage 1 checkpoint ───────────────────────────────────────────────
    ckpt_s1 = Path(s1_dir) / f"s1_{backbone_name}.pth"
    if not ckpt_s1.exists():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint Stage 1: {ckpt_s1}\n"
            f"Chạy train_stage1.py trước."
        )
    ckpt_data = torch.load(ckpt_s1, map_location=device)
    enc_dim   = ckpt_data["enc_dim"]
    feat_dim  = ckpt_data["feat_dim"]
    print(f"  Loaded S1 checkpoint: {ckpt_s1}")
    # ── Rebuild model + load encoder weights ─────────────────────────────────
    encoder, _, get_pg = build_backbone(backbone_name)
    model = SupConModel(encoder, enc_dim, feat_dim).to(device)
    model.load_state_dict(ckpt_data["model_state"])
    print(f"  enc_dim={enc_dim}  feat_dim={feat_dim}")
    # Freeze encoder
    for p in model.encoder.parameters():
        p.requires_grad = False
    model.encoder.eval()
    # ── Classifier head ───────────────────────────────────────────────────────
    classifier = nn.Linear(enc_dim, NUM_CLASSES).to(device)
    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_set = datasets.ImageFolder(train_dir, transform=get_transform_s2())
    val_set   = datasets.ImageFolder(val_dir,   transform=get_transform_eval())
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)
    print(f"  Train: {len(train_set)} ảnh | Val: {len(val_set)} ảnh")
    print(f"  Classes: {train_set.class_to_idx}")
    # ── Class weights ─────────────────────────────────────────────────────────
    labels_train  = [s[1] for s in train_set.samples]
    class_weights = compute_class_weights(labels_train, NUM_CLASSES, device)
    print(f"  Class weights: NORMAL={class_weights[0]:.4f}  PNEUMONIA={class_weights[1]:.4f}")
    # ── Loss + Optimizer ──────────────────────────────────────────────────────
    criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=class_weights)
    optimizer = optim.Adam(classifier.parameters(), lr=LR_CLF)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler()
    es     = EarlyStopping(patience=ES_PATIENCE, mode="max")
    loss_hist    = []
    val_f1_hist  = []
    best_f1      = 0.0
    best_epoch   = 0
    best_clf_st  = None
    best_mod_st  = None
    print(f"\n  Config: max {EPOCHS} ep | FocalLoss γ={FOCAL_GAMMA} | EarlyStop patience={ES_PATIENCE}")
    if unfreeze_after > 0:
        print(f"  Unfreeze encoder sau epoch {unfreeze_after}")
    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        # Unfreeze encoder sau N epoch (full fine-tuning)
        if unfreeze_after > 0 and epoch == unfreeze_after + 1:
            print(f"\n  [Ep {epoch}] Unfreeze encoder → full fine-tuning")
            for p in model.encoder.parameters():
                p.requires_grad = True
            model.encoder.train()
            enc_groups = get_pg(model.encoder, LR_ENC_FT)
            for g in enc_groups:
                optimizer.add_param_group(g)
        model.train() if (unfreeze_after > 0 and epoch > unfreeze_after) else model.eval()
        classifier.train()
        running = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            with torch.cuda.amp.autocast():
                if unfreeze_after > 0 and epoch > unfreeze_after:
                    feats  = model.encoder(images).flatten(1)
                else:
                    with torch.no_grad():
                        feats = model.encoder(images).flatten(1)
                logits = classifier(feats)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            running += loss.item()
        scheduler.step()
        avg_loss = running / len(train_loader)
        loss_hist.append(avg_loss)
        # Validate
        val_acc, val_f1, _, _ = evaluate(model, classifier, val_loader, device)
        val_f1_hist.append(val_f1)
        if val_f1 > best_f1:
            best_f1     = val_f1
            best_epoch  = epoch
            best_clf_st = {k: v.clone() for k, v in classifier.state_dict().items()}
            best_mod_st = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch <= 3 or epoch % 5 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Ep[{epoch:3d}/{EPOCHS}]  "
                  f"Loss={avg_loss:.4f}  ValAcc={val_acc:.4f}  ValF1={val_f1:.4f}  "
                  f"LR={lr_now:.2e}  {'★' if epoch == best_epoch else ''}")
        if es(val_f1):
            print(f"\n  EarlyStopping tại epoch {epoch} (best F1={best_f1:.4f} @ ep{best_epoch})")
            break
    # ── Lưu best checkpoint ───────────────────────────────────────────────────
    ckpt_out = Path(out_dir) / f"best_{backbone_name}.pth"
    torch.save({
        "backbone":     backbone_name,
        "enc_dim":      enc_dim,
        "feat_dim":     feat_dim,
        "model_state":  best_mod_st,
        "clf_state":    best_clf_st,
        "class_weights": class_weights.cpu().tolist(),
        "val_f1_best":  best_f1,
        "best_epoch":   best_epoch,
        "loss_s2":      loss_hist,
        "val_f1_hist":  val_f1_hist,
    }, ckpt_out)
    elapsed = (time.time() - t0) / 60
    print(f"\n  ✓ Saved: {ckpt_out}")
    print(f"  Best Val F1-macro: {best_f1:.4f} @ epoch {best_epoch}")
    print(f"  Thời gian: {elapsed:.1f} phút")
    del criterion, optimizer, scheduler, scaler, model, classifier
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"backbone": backbone_name, "val_f1": best_f1, "best_epoch": best_epoch}
# ─── Main ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Stage 2: Classifier Fine-tuning")
    p.add_argument("--s1-dir",    required=True,
                   help="Thư mục chứa checkpoint Stage 1 (s1_*.pth)")
    p.add_argument("--train-dir", required=True,
                   help="Thư mục tập train (ImageFolder)")
    p.add_argument("--val-dir",   required=True,
                   help="Thư mục tập val (ImageFolder)")
    p.add_argument("--out-dir",   default="outputs/stage2",
                   help="Thư mục lưu best_*.pth (default: outputs/stage2)")
    p.add_argument("--backbone",  default=None, choices=SUPPORTED_BACKBONES,
                   help="Chỉ fine-tune một backbone (mặc định: tất cả)")
    p.add_argument("--epochs",         type=int,   default=EPOCHS,    help=f"Max epoch (default: {EPOCHS})")
    p.add_argument("--batch-size",     type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",             type=float, default=LR_CLF,    help="LR classifier")
    p.add_argument("--patience",       type=int,   default=ES_PATIENCE)
    p.add_argument("--unfreeze-after", type=int,   default=0,
                   help="Unfreeze encoder sau N epoch (0 = giữ frozen hoàn toàn)")
    return p.parse_args()
def main():
    args = parse_args()
    global EPOCHS, BATCH_SIZE, LR_CLF, ES_PATIENCE
    EPOCHS      = args.epochs
    BATCH_SIZE  = args.batch_size
    LR_CLF      = args.lr
    ES_PATIENCE = args.patience
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
    torch.manual_seed(SEED)
    backbones = [args.backbone] if args.backbone else SUPPORTED_BACKBONES
    results   = []
    for name in backbones:
        r = train_one(
            backbone_name  = name,
            s1_dir         = args.s1_dir,
            train_dir      = args.train_dir,
            val_dir        = args.val_dir,
            out_dir        = args.out_dir,
            device         = device,
            unfreeze_after = args.unfreeze_after,
        )
        results.append(r)
    # Summary
    print(f"\n{'='*65}")
    print("  STAGE 2 – TÓM TẮT KẾT QUẢ VAL")
    print(f"{'='*65}")
    print(f"  {'Backbone':<20} {'Val F1-macro':>13} {'Best Epoch':>11}")
    print(f"  {'-'*46}")
    for r in results:
        print(f"  {r['backbone']:<20} {r['val_f1']:>12.4f} {r['best_epoch']:>11}")
    best = max(results, key=lambda x: x["val_f1"])
    print(f"\n  Best: {best['backbone']}  F1={best['val_f1']:.4f}")
if __name__ == "__main__":
    main()
