"""
evaluate.py – Final Evaluation trên Test Set
=============================================
Load best checkpoint từ Stage 2, chạy inference trên Test set,
in ra đầy đủ các chỉ số phân loại.
Metrics được tính:
  - Accuracy           (ngưỡng 0.5 và optimal threshold)
  - Precision / Recall / F1 per class
  - F1-macro           (primary metric)
  - ROC-AUC
  - Optimal Threshold  (Youden's J statistic: argmax(TPR - FPR))
  - Recall NORMAL / Recall PNEUMONIA (đặc biệt quan trọng trong y tế)
Cách dùng:
  # Evaluate tất cả backbone
  python evaluate.py \\
      --s2-dir outputs/stage2 \\
      --test-dir /path/test   \\
      --out-dir  outputs/eval
  # Evaluate một backbone cụ thể
  python evaluate.py --backbone densenet121 ...
  # Lưu confusion matrix và ROC curve
  python evaluate.py ... --plot
Output:
  outputs/eval/
  ├── eval_{backbone}_report.txt   ← classification report đầy đủ
  └── eval_{backbone}_plots.png    ← confusion matrix + ROC (nếu --plot)
"""
import os
import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.backbone import build_backbone, SupConModel, SUPPORTED_BACKBONES
from src.data.transforms import get_transform_eval
# ─── Constants ────────────────────────────────────────────────────────────────
NUM_CLASSES  = 2
CLASS_NAMES  = ["NORMAL", "PNEUMONIA"]
BATCH_SIZE   = 64
# ─── Inference ────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(
    model: SupConModel,
    classifier: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chạy inference trên toàn bộ loader.
    Returns:
        probs  : [N] – xác suất lớp PNEUMONIA (class index 1)
        preds  : [N] – predicted labels (threshold=0.5)
        labels : [N] – ground truth labels
    """
    model.eval()
    classifier.eval()
    all_probs, all_preds, all_labels = [], [], []
    for images, labels in loader:
        images = images.to(device)
        with torch.cuda.amp.autocast():
            feats  = model.encoder(images).flatten(1)
            logits = classifier(feats)
        probs = torch.softmax(logits, dim=1)[:, 1]   # prob PNEUMONIA
        preds = logits.argmax(dim=1)
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_preds), np.array(all_labels)
# ─── Optimal threshold ────────────────────────────────────────────────────────
def find_optimal_threshold(
    labels: np.ndarray, probs: np.ndarray
) -> tuple[float, np.ndarray]:
    """
    Tìm threshold tối ưu theo Youden's J: argmax(TPR - FPR).
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx]), (fpr, tpr)
# ─── Compute metrics ──────────────────────────────────────────────────────────
def compute_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    threshold: float | None = None,
) -> dict:
    """Tính toàn bộ metrics với threshold mặc định và optimal."""
    # Default threshold (0.5)
    acc_05 = accuracy_score(labels, preds)
    f1_05  = f1_score(labels, preds, average="macro", zero_division=0)
    auc    = roc_auc_score(labels, probs)
    # Optimal threshold
    opt_thr, (fpr, tpr) = find_optimal_threshold(labels, probs)
    preds_opt  = (probs >= opt_thr).astype(int)
    acc_opt    = accuracy_score(labels, preds_opt)
    f1_opt     = f1_score(labels, preds_opt, average="macro", zero_division=0)
    prec_opt   = precision_score(labels, preds_opt, average="macro", zero_division=0)
    rec_opt    = recall_score(labels, preds_opt, average="macro", zero_division=0)
    rec_normal = recall_score(labels, preds_opt, labels=[0], average="micro", zero_division=0)
    rec_pneumo = recall_score(labels, preds_opt, labels=[1], average="micro", zero_division=0)
    prec_n     = precision_score(labels, preds_opt, labels=[0], average="micro", zero_division=0)
    prec_p     = precision_score(labels, preds_opt, labels=[1], average="micro", zero_division=0)
    return {
        "acc_05":     acc_05,
        "f1_05":      f1_05,
        "acc_opt":    acc_opt,
        "f1_opt":     f1_opt,
        "precision":  prec_opt,
        "recall":     rec_opt,
        "auc":        auc,
        "opt_thr":    opt_thr,
        "rec_normal": rec_normal,
        "rec_pneumo": rec_pneumo,
        "prec_normal":prec_n,
        "prec_pneumo":prec_p,
        "fpr":        fpr.tolist(),
        "tpr":        tpr.tolist(),
    }
# ─── Print report ─────────────────────────────────────────────────────────────
def print_report(backbone: str, m: dict, labels: np.ndarray, preds_opt: np.ndarray) -> str:
    """In và trả về chuỗi report."""
    sep = "=" * 64
    lines = [
        sep,
        f"EVALUATION REPORT – {backbone.upper()}",
        sep,
        f"  AUC          : {m['auc']:.4f}",
        "",
        f"  ── Threshold = 0.50 (default) ──────────────────────────",
        f"  Accuracy     : {m['acc_05']:.4f}  ({m['acc_05']*100:.2f}%)",
        f"  F1-macro     : {m['f1_05']:.4f}",
        "",
        f"  ── Threshold = {m['opt_thr']:.3f} (optimal, Youden's J) ──────",
        f"  Accuracy     : {m['acc_opt']:.4f}  ({m['acc_opt']*100:.2f}%)",
        f"  F1-macro     : {m['f1_opt']:.4f}",
        f"  Precision    : {m['precision']:.4f}",
        f"  Recall       : {m['recall']:.4f}",
        "",
        f"  ── Per-class (optimal threshold) ────────────────────────",
        f"  {'Class':<12} {'Precision':>10} {'Recall':>10}",
        f"  {'-'*34}",
        f"  {'NORMAL':<12} {m['prec_normal']:>10.4f} {m['rec_normal']:>10.4f}",
        f"  {'PNEUMONIA':<12} {m['prec_pneumo']:>10.4f} {m['rec_pneumo']:>10.4f}",
        "",
        "  ── Sklearn Classification Report (optimal thr) ──────────",
        classification_report(labels, preds_opt, target_names=CLASS_NAMES),
        sep,
    ]
    report = "\n".join(lines)
    print(report)
    return report
# ─── Plots ────────────────────────────────────────────────────────────────────
def plot_results(
    backbone: str,
    labels: np.ndarray,
    preds_opt: np.ndarray,
    m: dict,
    out_path: Path,
) -> None:
    """Vẽ Confusion Matrix + ROC Curve."""
    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig)
    fig.suptitle(f"Evaluation – {backbone.upper()}", fontsize=14, fontweight="bold")
    # Confusion Matrix
    ax_cm = fig.add_subplot(gs[0])
    cm    = confusion_matrix(labels, preds_opt)
    im    = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(CLASS_NAMES); ax_cm.set_yticklabels(CLASS_NAMES)
    ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("True")
    ax_cm.set_title(f"Confusion Matrix\n(thr={m['opt_thr']:.3f})")
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                       fontsize=14, color="white" if cm[i,j] > cm.max() / 2 else "black")
    # ROC Curve
    ax_roc = fig.add_subplot(gs[1])
    ax_roc.plot(m["fpr"], m["tpr"], color="#5FA8D3", lw=2,
                label=f"AUC={m['auc']:.4f}")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {out_path}")
# ─── Evaluate one backbone ────────────────────────────────────────────────────
def evaluate_one(
    backbone_name: str,
    s2_dir: str,
    test_dir: str,
    out_dir: str,
    device: torch.device,
    plot: bool = False,
) -> dict:
    """Load best checkpoint và evaluate trên test set."""
    print(f"\n{'='*65}")
    print(f"  EVALUATE | BACKBONE: {backbone_name.upper()}")
    print(f"{'='*65}")
    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt_path = Path(s2_dir) / f"best_{backbone_name}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint Stage 2: {ckpt_path}\n"
            f"Chạy train_stage2.py trước."
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"  Loaded: {ckpt_path}")
    print(f"  Val F1 lúc training: {ckpt.get('val_f1_best', 'N/A'):.4f}")
    enc_dim  = ckpt["enc_dim"]
    feat_dim = ckpt["feat_dim"]
    # ── Rebuild model ─────────────────────────────────────────────────────────
    encoder, _, _ = build_backbone(backbone_name)
    model         = SupConModel(encoder, enc_dim, feat_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    classifier = nn.Linear(enc_dim, NUM_CLASSES).to(device)
    classifier.load_state_dict(ckpt["clf_state"])
    # ── DataLoader (test set) ─────────────────────────────────────────────────
    test_set    = datasets.ImageFolder(test_dir, transform=get_transform_eval())
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=2, pin_memory=True)
    print(f"  Test set: {len(test_set)} ảnh | Classes: {test_set.class_to_idx}")
    # ── Inference ─────────────────────────────────────────────────────────────
    probs, preds, labels = run_inference(model, classifier, test_loader, device)
    # ── Metrics ───────────────────────────────────────────────────────────────
    m         = compute_metrics(labels, probs, preds)
    preds_opt = (probs >= m["opt_thr"]).astype(int)
    report    = print_report(backbone_name, m, labels, preds_opt)
    # ── Save report ───────────────────────────────────────────────────────────
    out = Path(out_dir)
    report_path = out / f"eval_{backbone_name}_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved report: {report_path}")
    # Save metrics as JSON
    m_json = {k: v for k, v in m.items() if k not in ("fpr", "tpr")}
    (out / f"eval_{backbone_name}_metrics.json").write_text(
        json.dumps(m_json, indent=2), encoding="utf-8"
    )
    # ── Plot ──────────────────────────────────────────────────────────────────
    if plot:
        plot_results(
            backbone_name, labels, preds_opt, m,
            out / f"eval_{backbone_name}_plots.png",
        )
    return {"backbone": backbone_name, **m_json}
# ─── Main ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate best model on Test set")
    p.add_argument("--s2-dir",   required=True,
                   help="Thư mục chứa checkpoint Stage 2 (best_*.pth)")
    p.add_argument("--test-dir", required=True,
                   help="Thư mục tập test (ImageFolder)")
    p.add_argument("--out-dir",  default="outputs/eval",
                   help="Thư mục lưu report và plot (default: outputs/eval)")
    p.add_argument("--backbone", default=None, choices=SUPPORTED_BACKBONES,
                   help="Chỉ evaluate một backbone (mặc định: tất cả)")
    p.add_argument("--plot", action="store_true",
                   help="Lưu confusion matrix và ROC curve")
    return p.parse_args()
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    backbones = [args.backbone] if args.backbone else SUPPORTED_BACKBONES
    results   = []
    for name in backbones:
        r = evaluate_one(
            backbone_name = name,
            s2_dir        = args.s2_dir,
            test_dir      = args.test_dir,
            out_dir       = args.out_dir,
            device        = device,
            plot          = args.plot,
        )
        results.append(r)
    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  KẾT QUẢ ĐÁNH GIÁ TRÊN TEST SET")
    print(f"{'='*72}")
    print(f"  {'Backbone':<20} {'Acc(opt)':>9} {'AUC':>7} {'F1-macro':>9} "
          f"{'Rec-N':>7} {'Rec-P':>7} {'Thr':>7}")
    print(f"  {'-'*64}")
    for r in results:
        print(f"  {r['backbone']:<20} "
              f"{r['acc_opt']*100:>8.2f}% "
              f"{r['auc']:>7.4f} "
              f"{r['f1_opt']:>9.4f} "
              f"{r['rec_normal']*100:>6.1f}% "
              f"{r['rec_pneumo']*100:>6.1f}% "
              f"{r['opt_thr']:>7.3f}")
    print(f"{'='*72}")
    best = max(results, key=lambda x: x["f1_opt"])
    print(f"\n  Best backbone (test F1-macro): {best['backbone'].upper()}  "
          f"F1={best['f1_opt']:.4f}  AUC={best['auc']:.4f}")
if __name__ == "__main__":
    main()
