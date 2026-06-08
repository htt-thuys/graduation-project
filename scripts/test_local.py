"""
test_local.py – Local Testing Script
====================================
Creates a minimal test dataset and runs all 3 stages (Stage 1, Stage 2, Evaluate)
with 5 epochs to verify the codebase works correctly.

Usage:
    python test_local.py
    
The script will:
1. Create test_data/ with fake chest X-ray images (50 NORMAL, 50 PNEUMONIA per split)
2. Run Stage 1 (SupCon pre-training) with 5 epochs
3. Run Stage 2 (classifier fine-tuning) with 5 epochs
4. Run Evaluate (test set evaluation)
5. Print summary results
"""
import os
import sys
import shutil
import time
import random
import numpy as np
from pathlib import Path
import torch
from PIL import Image

# Configuration
TEST_DATA_DIR = Path("test_data")
OUTPUT_DIR = Path("test_output")
TRAIN_DIR = TEST_DATA_DIR / "train"
VAL_DIR = TEST_DATA_DIR / "val"
TEST_DIR = TEST_DATA_DIR / "test"

STAGE1_OUT = OUTPUT_DIR / "stage1"
STAGE2_OUT = OUTPUT_DIR / "stage2"
EVAL_OUT = OUTPUT_DIR / "eval"

N_IMAGES_PER_CLASS = 20  # Small number for quick testing
EPOCHS_TEST = 5
IMG_SIZE = 224


def create_fake_image(size: int = 224) -> Image.Image:
    """Create a fake X-ray image (random grayscale)."""
    arr = np.random.randint(0, 256, (size, size), dtype=np.uint8)
    return Image.fromarray(arr, mode='L')


def setup_test_data():
    """Create minimal test dataset structure."""
    print("\n" + "="*65)
    print("  SETTING UP TEST DATA")
    print("="*65)
    
    # Clean up if exists
    if TEST_DATA_DIR.exists():
        shutil.rmtree(TEST_DATA_DIR)
    
    # Create directories
    for split in ["train", "val", "test"]:
        for cls in ["NORMAL", "PNEUMONIA"]:
            (TEST_DATA_DIR / split / cls).mkdir(parents=True, exist_ok=True)
    
    # Create fake images
    print(f"\n  Creating fake images ({N_IMAGES_PER_CLASS} per class per split)...")
    for split in ["train", "val", "test"]:
        for cls in ["NORMAL", "PNEUMONIA"]:
            cls_dir = TEST_DATA_DIR / split / cls
            for i in range(N_IMAGES_PER_CLASS):
                img = create_fake_image(IMG_SIZE)
                img_path = cls_dir / f"{cls.lower()}_{i:03d}.jpeg"
                img.save(img_path)
            print(f"    ✓ {split:5s}/{cls:10s}: {N_IMAGES_PER_CLASS} images")
    
    # Print summary
    print(f"\n  Dataset structure:")
    for split in ["train", "val", "test"]:
        n_normal = len(list((TEST_DATA_DIR / split / "NORMAL").glob("*.jpeg")))
        n_pneumonia = len(list((TEST_DATA_DIR / split / "PNEUMONIA").glob("*.jpeg")))
        total = n_normal + n_pneumonia
        print(f"    {split:5s}: NORMAL={n_normal:3d}  PNEUMONIA={n_pneumonia:3d}  Total={total}")


def run_stage1():
    """Run Stage 1: SupCon Pre-training with 5 epochs."""
    print("\n" + "="*65)
    print("  STAGE 1: SUPCON PRE-TRAINING")
    print("="*65)
    print(f"\n  Training parameters:")
    print(f"    Epochs: {EPOCHS_TEST}")
    print(f"    Batch size: 16 (reduced for test)")
    print(f"    Train dir: {TRAIN_DIR}")
    print(f"    Output dir: {STAGE1_OUT}")
    
    STAGE1_OUT.mkdir(parents=True, exist_ok=True)
    
    cmd = (
        f'"{sys.executable}" tools/train_stage1.py '
        f"--train-dir {TRAIN_DIR} "
        f"--out-dir {STAGE1_OUT} "
        f"--epochs {EPOCHS_TEST} "
        f"--batch-size 16"
    )
    
    print(f"\n  Running: {cmd}\n")
    ret = os.system(cmd)
    
    if ret != 0:
        print("\n  ❌ Stage 1 FAILED")
        return False
    
    print("\n  ✓ Stage 1 completed successfully")
    
    # Verify checkpoint files exist
    ckpts = list(STAGE1_OUT.glob("s1_*.pth"))
    print(f"\n  Checkpoints created: {len(ckpts)}")
    for ckpt in sorted(ckpts):
        size_mb = ckpt.stat().st_size / 1e6
        print(f"    - {ckpt.name} ({size_mb:.1f} MB)")
    
    return True


def run_stage2():
    """Run Stage 2: Classifier Fine-tuning with 5 epochs."""
    print("\n" + "="*65)
    print("  STAGE 2: CLASSIFIER FINE-TUNING")
    print("="*65)
    print(f"\n  Training parameters:")
    print(f"    Max epochs: {EPOCHS_TEST}")
    print(f"    Batch size: 32")
    print(f"    Train dir: {TRAIN_DIR}")
    print(f"    Val dir: {VAL_DIR}")
    print(f"    S1 dir: {STAGE1_OUT}")
    print(f"    Output dir: {STAGE2_OUT}")
    
    STAGE2_OUT.mkdir(parents=True, exist_ok=True)
    
    cmd = (
        f'"{sys.executable}" tools/train_stage2.py '
        f"--s1-dir {STAGE1_OUT} "
        f"--train-dir {TRAIN_DIR} "
        f"--val-dir {VAL_DIR} "
        f"--out-dir {STAGE2_OUT} "
        f"--epochs {EPOCHS_TEST} "
        f"--batch-size 32"
    )
    
    print(f"\n  Running: {cmd}\n")
    ret = os.system(cmd)
    
    if ret != 0:
        print("\n  ❌ Stage 2 FAILED")
        return False
    
    print("\n  ✓ Stage 2 completed successfully")
    
    # Verify checkpoint files exist
    ckpts = list(STAGE2_OUT.glob("best_*.pth"))
    print(f"\n  Checkpoints created: {len(ckpts)}")
    for ckpt in sorted(ckpts):
        size_mb = ckpt.stat().st_size / 1e6
        print(f"    - {ckpt.name} ({size_mb:.1f} MB)")
    
    return True


def run_evaluate():
    """Run Evaluate: Test set evaluation."""
    print("\n" + "="*65)
    print("  EVALUATE: TEST SET EVALUATION")
    print("="*65)
    print(f"\n  Evaluation parameters:")
    print(f"    S2 dir: {STAGE2_OUT}")
    print(f"    Test dir: {TEST_DIR}")
    print(f"    Output dir: {EVAL_OUT}")
    
    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    
    cmd = (
        f'"{sys.executable}" tools/evaluate.py '
        f"--s2-dir {STAGE2_OUT} "
        f"--test-dir {TEST_DIR} "
        f"--out-dir {EVAL_OUT} "
        f"--plot"
    )
    
    print(f"\n  Running: {cmd}\n")
    ret = os.system(cmd)
    
    if ret != 0:
        print("\n  ❌ Evaluation FAILED")
        return False
    
    print("\n  ✓ Evaluation completed successfully")
    
    # List output files
    reports = list(EVAL_OUT.glob("*.txt"))
    plots = list(EVAL_OUT.glob("*.png"))
    
    if reports:
        print(f"\n  Reports created: {len(reports)}")
        for report in sorted(reports):
            print(f"    - {report.name}")
    
    if plots:
        print(f"\n  Plots created: {len(plots)}")
        for plot in sorted(plots):
            size_mb = plot.stat().st_size / 1e6
            print(f"    - {plot.name} ({size_mb:.1f} MB)")
    
    return True


def print_summary():
    """Print final summary."""
    print("\n" + "="*65)
    print("  TEST SUMMARY")
    print("="*65)
    
    print(f"\n  Test data: {TEST_DATA_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    
    print(f"\n  Stage 1 (SupCon) Checkpoints:")
    s1_ckpts = list(STAGE1_OUT.glob("s1_*.pth"))
    for ckpt in sorted(s1_ckpts):
        print(f"    ✓ {ckpt.name}")
    
    print(f"\n  Stage 2 (Classifier) Checkpoints:")
    s2_ckpts = list(STAGE2_OUT.glob("best_*.pth"))
    for ckpt in sorted(s2_ckpts):
        print(f"    ✓ {ckpt.name}")
    
    print(f"\n  Evaluation Results:")
    reports = list(EVAL_OUT.glob("*_report.txt"))
    for report in sorted(reports):
        print(f"    ✓ {report.name}")
        # Print first few lines of report
        with open(report, 'r') as f:
            lines = f.readlines()[:15]
            for line in lines:
                print(f"      {line.rstrip()}")


def main():
    print(f"\n{'='*65}")
    print("  CODEBASE LOCAL TEST (5 EPOCHS)")
    print(f"{'='*65}")
    print(f"  GPU: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Device: {torch.cuda.get_device_name(0)}")
    
    # Check dependencies
    print(f"\n  Checking dependencies...")
    try:
        import torchvision
        import sklearn
        import pandas
        from PIL import Image
        print("    ✓ All required packages available")
    except ImportError as e:
        print(f"    ❌ Missing package: {e}")
        print("\n  Install with: pip install torch torchvision scikit-learn pandas pillow tqdm matplotlib seaborn")
        return False
    
    # Step 1: Create test data
    print(f"\n  Step 1: Creating test dataset...")
    setup_test_data()
    
    # Step 2: Run Stage 1
    print(f"\n  Step 2: Running Stage 1 (SupCon Pre-training)...")
    if not run_stage1():
        return False
    
    # Step 3: Run Stage 2
    print(f"\n  Step 3: Running Stage 2 (Classifier Fine-tuning)...")
    if not run_stage2():
        return False
    
    # Step 4: Run Evaluate
    print(f"\n  Step 4: Running Evaluate (Test Set Evaluation)...")
    if not run_evaluate():
        return False
    
    # Step 5: Summary
    print_summary()
    
    print(f"\n{'='*65}")
    print("  ✅ ALL TESTS PASSED")
    print(f"{'='*65}")
    print(f"\n  Test completed successfully!")
    print(f"  Output directory: {OUTPUT_DIR.absolute()}")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
