"""
models/transforms.py
====================
Data augmentation và preprocessing transforms.
Transforms được định nghĩa:
  - TwoCropTransform   : wrap một transform để sinh 2 view (positive pair cho SupCon)
  - get_transform_s1() : augmentation mạnh cho Stage 1 (SupCon pre-training)
  - get_transform_s2() : augmentation nhẹ cho Stage 2 (classifier fine-tuning)
  - get_transform_eval(): transform chuẩn cho validation và test (không augment)
ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
"""
import torchvision.transforms as T
# ─── Constants ────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE_S1   = 160   # ảnh nhỏ hơn trong S1 để tăng tốc + augment đa dạng hơn
IMG_SIZE_S2   = 224   # ảnh chuẩn 224×224 cho S2 và eval
# ─── TwoCropTransform ─────────────────────────────────────────────────────────
class TwoCropTransform:
    """
    Áp dụng cùng một transform hai lần độc lập lên cùng một ảnh.
    Kết quả: [view1, view2] – hai augmentation khác nhau của cùng ảnh gốc.
    Hai views này là positive pair trong SupCon Loss.
    """
    def __init__(self, transform: T.Compose):
        self.transform = transform
    def __call__(self, x):
        return [self.transform(x), self.transform(x)]
# ─── Stage 1 transform ────────────────────────────────────────────────────────
def get_transform_s1() -> T.Compose:
    """
    Augmentation pipeline cho Stage 1 (SupCon pre-training).
    - Resize nhỏ hơn (160×160) → nhanh hơn, augment đa dạng hơn
    - Augmentation mạnh để học representation robust:
        RandomHorizontalFlip, RandomRotation, ColorJitter
    """
    return T.Compose([
        T.Resize((IMG_SIZE_S1, IMG_SIZE_S1)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
# ─── Stage 2 transform ────────────────────────────────────────────────────────
def get_transform_s2() -> T.Compose:
    """
    Augmentation pipeline cho Stage 2 (classifier training).
    - Resize 224×224 (standard)
    - Augmentation nhẹ hơn S1 (encoder đã frozen)
    """
    return T.Compose([
        T.Resize((IMG_SIZE_S2, IMG_SIZE_S2)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=5),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
# ─── Evaluation transform ─────────────────────────────────────────────────────
def get_transform_eval() -> T.Compose:
    """
    Transform chuẩn cho validation và test set.
    KHÔNG augment – chỉ resize và normalize.
    """
    return T.Compose([
        T.Resize((IMG_SIZE_S2, IMG_SIZE_S2)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])