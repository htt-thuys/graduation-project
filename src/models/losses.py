"""
models/losses.py
================
Định nghĩa các hàm loss dùng trong pipeline:
1. SupConLossWithQueue
   - Supervised Contrastive Loss (Khosla et al., 2020)
   - Cải tiến: Feature Queue (kích thước K) để tăng số negative samples
   - Effective negatives/anchor ≈ batch_size + K (≈ 2080 mặc định)
   - Formula:
       L = -Σ_{i} 1/|P(i)| × Σ_{p∈P(i)} log[
             exp(z_i·z_p / τ) /
             Σ_{a∈A(i)} exp(z_i·z_a / τ)
           ]
2. FocalLoss
   - Focal Loss (Lin et al., 2017) cho phân loại mất cân bằng
   - Formula: L = Σ (1 - p_t)^γ × CE(y, ŷ)
   - Dùng class_weight để tăng trọng số lớp thiểu số (NORMAL)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLossWithQueue(nn.Module):
    """
    Supervised Contrastive Loss + Feature Queue.
    Queue lưu các key features từ batch trước để làm additional negatives.
    Nhờ đó, số lượng negative samples hiệu quả tăng từ batch_size lên
    batch_size + queue_size (~2080 với batch=32, queue=2048).
    Args:
        temperature : nhiệt độ τ, kiểm soát độ sắc nét của phân phối (default: 0.3)
        queue_size  : số lượng key features lưu trong queue (default: 2048)
        feat_dim    : chiều của feature vector (default: 128)
    """
    def __init__(
        self,
        temperature: float = 0.3,
        queue_size: int = 2048,
        feat_dim: int = 128,
    ):
        super().__init__()
        self.temperature = temperature
        self.queue_size  = queue_size
        # Queue: [feat_dim, queue_size] – khởi tạo ngẫu nhiên, L2-normalized
        self.register_buffer(
            "queue",
            F.normalize(torch.randn(feat_dim, queue_size), dim=0)
        )
        self.register_buffer(
            "queue_labels",
            torch.zeros(queue_size, dtype=torch.long)
        )
        self.register_buffer(
            "queue_ptr",
            torch.zeros(1, dtype=torch.long)
        )
    @torch.no_grad()
    def _enqueue(self, keys: torch.Tensor, labels: torch.Tensor) -> None:
        """Thêm keys mới vào queue (FIFO, circular buffer)."""
        B   = keys.shape[0]
        ptr = int(self.queue_ptr)
        end = min(ptr + B, self.queue_size)
        n   = end - ptr
        self.queue[:, ptr:end]     = keys[:n].T.detach()
        self.queue_labels[ptr:end] = labels[:n].detach()
        self.queue_ptr[0]          = end % self.queue_size
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Tính SupCon Loss cho một batch gồm 2 view (TwoCropTransform).
        Args:
            features : [2B, feat_dim] – L2-normalized, gồm query ([:B]) và key ([B:])
            labels   : [B]            – nhãn lớp của batch
        Returns:
            loss : scalar tensor
        """
        device = features.device
        B      = features.shape[0] // 2
        q, k   = features[:B], features[B:]   # query, key
        # Concatenate key batch + queue → tất cả negatives
        all_keys   = torch.cat([k.T, self.queue.clone().detach()], dim=1)   # [feat_dim, B+K]
        all_labels = torch.cat([
            labels,
            self.queue_labels.clone().detach().to(device)
        ])   # [B+K]
        # Logits: [B, B+K]
        logits     = torch.matmul(q, all_keys) / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits     = logits - logits_max.detach()   # stability
        # Positive mask: same label as query
        pos_mask = torch.eq(
            labels.view(-1, 1), all_labels.view(1, -1)
        ).float().to(device)   # [B, B+K]
        # SupCon loss
        exp_logits = torch.exp(logits)
        log_prob   = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1).clamp(min=1)
        loss = -mean_log_prob_pos.mean()
        # Cập nhật queue
        self._enqueue(k, labels)
        return loss
# ─── Focal Loss ───────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss cho phân loại nhị phân / đa lớp với mất cân bằng.
    Formula: L = Σ weight[y] × (1 - p_t)^γ × CE(y, ŷ)
    Args:
        gamma       : focusing parameter (default: 2.0)
        weight      : class weights tensor [num_classes] (default: None)
        reduction   : 'mean' | 'sum' | 'none' (default: 'mean')
    """
    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma     = gamma
        self.weight    = weight
        self.reduction = reduction
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input  : [B, num_classes] – raw logits (chưa qua softmax)
            target : [B]              – ground truth class indices
        Returns:
            loss : scalar (nếu reduction='mean'/'sum') hoặc [B]
        """
        # Cross-entropy per sample
        ce_loss = F.cross_entropy(
            input, target,
            weight=self.weight.to(input.device) if self.weight is not None else None,
            reduction="none",
        )
        # p_t: xác suất của lớp đúng
        p_t     = torch.exp(-ce_loss)
        fl_loss = (1 - p_t) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return fl_loss.mean()
        elif self.reduction == "sum":
            return fl_loss.sum()
        return fl_loss
def compute_class_weights(
    labels: list[int],
    num_classes: int = 2,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Tính class weights theo công thức:
        weight[c] = total / (num_classes × count[c])
    Args:
        labels      : list các nhãn trong tập train
        num_classes : số lớp
        device      : thiết bị để đặt tensor
    Returns:
        weights : tensor [num_classes]
    """
    from collections import Counter
    count  = Counter(labels)
    total  = len(labels)
    weight = torch.tensor(
        [total / (num_classes * count.get(c, 1)) for c in range(num_classes)],
        dtype=torch.float32,
        device=device,
    )
    return weight
