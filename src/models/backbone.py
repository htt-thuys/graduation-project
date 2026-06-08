"""
models/backbone.py
==================
Factory function build_backbone() và class SupConModel.
Backbone được hỗ trợ:
  - resnet50       (ImageNet pretrained, enc_dim=2048)
  - densenet121    (ImageNet pretrained, enc_dim=1024)
  - efficientnet_b4 (ImageNet pretrained, enc_dim=1792)
Discriminative Learning Rate:
  Encoder được chia thành 3 nhóm layer với LR giảm dần từ cuối về đầu:
    Layer đầu  (frozen-like) → LR × 0.01
    Layer giữa              → LR × 0.10
    Layer cuối (fine-tune)  → LR × 0.50
  Projection Head / Classifier → LR × 1.00 (full LR)
SupConModel:
  Input  → Encoder → flatten → [B, enc_dim]
         → Projection Head → L2-normalize → [B, feat_dim]
  .get_features(x) trả về raw encoder output (dùng cho classifier ở Stage 2)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
# ─── Encoder configurations ───────────────────────────────────────────────────
_BACKBONE_CONFIGS = {
    "resnet50": {
        "weights": models.ResNet50_Weights.IMAGENET1K_V1,
        "enc_dim": 2048,
    },
    "densenet121": {
        "weights": models.DenseNet121_Weights.IMAGENET1K_V1,
        "enc_dim": 1024,
    },
    "efficientnet_b4": {
        "weights": models.EfficientNet_B4_Weights.IMAGENET1K_V1,
        "enc_dim": 1792,
    },
}
SUPPORTED_BACKBONES = list(_BACKBONE_CONFIGS.keys())
def _build_resnet50():
    bb  = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    enc = nn.Sequential(*list(bb.children())[:-1])   # output: [B, 2048, 1, 1]
    def get_param_groups(encoder: nn.Module, base_lr: float) -> list[dict]:
        ch = list(encoder.children())
        return [
            {"params": [p for m in ch[:5]  for p in m.parameters()], "lr": base_lr * 0.01},
            {"params": [p for m in ch[5:7] for p in m.parameters()], "lr": base_lr * 0.10},
            {"params": [p for m in ch[7:]  for p in m.parameters()], "lr": base_lr * 0.50},
        ]
    return enc, 2048, get_param_groups
def _build_densenet121():
    bb  = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    enc = nn.Sequential(
        bb.features,
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
    )   # output: [B, 1024, 1, 1]
    def get_param_groups(encoder: nn.Module, base_lr: float) -> list[dict]:
        feats = list(encoder[0].children())
        return [
            {"params": [p for m in feats[:4]  for p in m.parameters()], "lr": base_lr * 0.01},
            {"params": [p for m in feats[4:6] for p in m.parameters()], "lr": base_lr * 0.10},
            {"params": [p for m in feats[6:]  for p in m.parameters()], "lr": base_lr * 0.50},
        ]
    return enc, 1024, get_param_groups
def _build_efficientnet_b4():
    bb  = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    enc = nn.Sequential(
        bb.features,
        nn.AdaptiveAvgPool2d((1, 1)),
    )   # output: [B, 1792, 1, 1]
    def get_param_groups(encoder: nn.Module, base_lr: float) -> list[dict]:
        feats = list(encoder[0].children())
        return [
            {"params": [p for m in feats[:3]  for p in m.parameters()], "lr": base_lr * 0.01},
            {"params": [p for m in feats[3:6] for p in m.parameters()], "lr": base_lr * 0.10},
            {"params": [p for m in feats[6:]  for p in m.parameters()], "lr": base_lr * 0.50},
        ]
    return enc, 1792, get_param_groups
_BUILDERS = {
    "resnet50":        _build_resnet50,
    "densenet121":     _build_densenet121,
    "efficientnet_b4": _build_efficientnet_b4,
}
def build_backbone(name: str) -> tuple:
    """
    Khởi tạo encoder từ backbone đã chọn.
    Returns:
        encoder          : nn.Module – phần encoder (không có head phân loại)
        enc_dim          : int       – số chiều output của encoder
        get_param_groups : callable  – trả về list[dict] tham số theo LR nhóm
    """
    if name not in _BUILDERS:
        raise ValueError(
            f"Backbone '{name}' không được hỗ trợ. "
            f"Chọn một trong: {SUPPORTED_BACKBONES}"
        )
    return _BUILDERS[name]()
# ─── SupConModel ──────────────────────────────────────────────────────────────
class SupConModel(nn.Module):
    """
    Encoder + Projection Head cho Supervised Contrastive Learning.
    Architecture:
      Input → Encoder → flatten [B, enc_dim]
            → Linear(enc_dim→512) → BN → ReLU → Linear(512→feat_dim)
            → L2-normalize → [B, feat_dim]
    """
    def __init__(self, encoder: nn.Module, encoder_dim: int, feat_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feat_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Trả về L2-normalized features cho SupCon Loss."""
        z = self.encoder(x).flatten(1)   # [B, enc_dim]
        z = self.head(z)                  # [B, feat_dim]
        return F.normalize(z, dim=1)
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Trả về raw encoder output (dùng cho classifier Stage 2)."""
        with torch.no_grad():
            return self.encoder(x).flatten(1)   # [B, enc_dim]