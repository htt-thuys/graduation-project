import os
import hashlib
import random
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

CLASS_NAMES = ['NORMAL', 'PNEUMONIA']
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
SEED        = 42
MIN_SIZE_PX = 64
MAX_RATIO   = 5.0

random.seed(SEED)
np.random.seed(SEED)

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Chest X-Ray Pneumonia dataset")
    parser.add_argument("--input", type=str, required=True, help="Thư mục chứa dataset gốc (có train, val, test)")
    parser.add_argument("--output", type=str, required=True, help="Thư mục xuất dataset đã chia")
    return parser.parse_args()


def collect_images(root_in: Path) -> list:
    records = []
    for split_name in ['train', 'val', 'test']:
        split_dir = root_in / split_name
        for cls in CLASS_NAMES:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                print(f"  [WARN] Không tìm thấy: {cls_dir}")
                continue
            for ext in ['*.jpeg', '*.jpg', '*.png', '*.JPEG', '*.JPG', '*.PNG']:
                for p in cls_dir.glob(ext):
                    records.append({
                        'path':           str(p),
                        'filename':       p.name,
                        'label':          cls,
                        'original_split': split_name,
                    })
    return records

def check_image(row: dict) -> dict:
    p = row['path']
    result = {
        'path':              p,
        'is_corrupt':        False,
        'width':             None,
        'height':            None,
        'mode':              None,
        'file_size_kb':      os.path.getsize(p) / 1024,
        'md5':               None,
        'is_too_small':      False,
        'is_abnormal_ratio': False,
        'is_grayscale':      False,
    }
    try:
        with Image.open(p) as img:
            img.verify()   
            
        with Image.open(p) as img:
            img_rgb = img.convert('RGB')
            w, h    = img_rgb.size
            result['width']  = w
            result['height'] = h
            result['mode']   = img.mode

            with open(p, 'rb') as f:
                result['md5'] = hashlib.md5(f.read()).hexdigest()

            result['is_too_small']      = min(w, h) < MIN_SIZE_PX
            result['is_abnormal_ratio'] = max(w / h, h / w) > MAX_RATIO

            arr = np.array(img_rgb)
            result['is_grayscale'] = bool(
                np.allclose(arr[:, :, 0], arr[:, :, 1], atol=5) and
                np.allclose(arr[:, :, 1], arr[:, :, 2], atol=5)
            )
    except (UnidentifiedImageError, Exception):
        result['is_corrupt'] = True
    return result

def preprocess_and_filter(df_raw: pd.DataFrame) -> pd.DataFrame:
    print('\nKiểm tra chất lượng ảnh...')
    quality_records = []
    for row in tqdm(df_raw.to_dict('records'), desc='Quality check'):
        quality_records.append(check_image(row))
        
    df_quality = pd.DataFrame(quality_records)
    df_merged  = df_raw.merge(df_quality, on='path')

    md5_counts = df_merged['md5'].value_counts()
    dup_md5    = set(md5_counts[md5_counts > 1].index)
    df_merged['is_duplicate'] = df_merged['md5'].isin(dup_md5)
    
    df_merged['keep_dup'] = ~df_merged.duplicated(subset=['md5'], keep='first')
    df_merged.loc[~df_merged['is_duplicate'], 'keep_dup'] = True

    n_corrupt      = df_merged['is_corrupt'].sum()
    n_too_small    = df_merged['is_too_small'].sum()
    n_abnormal     = df_merged['is_abnormal_ratio'].sum()
    n_dup          = df_merged['is_duplicate'].sum()
    n_grayscale    = df_merged['is_grayscale'].sum()

    print(f'\n── Báo cáo chất lượng ──────────────────────────────')
    print(f'  Tổng ảnh              : {len(df_merged)}')
    print(f'  Corrupt               : {n_corrupt}')
    print(f'  Quá nhỏ (<{MIN_SIZE_PX}px)      : {n_too_small}')
    print(f'  Tỉ lệ bất thường      : {n_abnormal}')
    print(f'  Duplicate (MD5)       : {n_dup}  → loại {n_dup - (df_merged["is_duplicate"] & ~df_merged["keep_dup"]).sum()} bản thừa')
    print(f'  Grayscale thực sự     : {n_grayscale} / {len(df_merged)} ({(n_grayscale/len(df_merged))*100:.1f}%)')
    
    mask_keep = (
        ~df_merged['is_corrupt'] &
        ~df_merged['is_too_small'] &
        ~df_merged['is_abnormal_ratio'] &
        df_merged['keep_dup']
    )
    df_clean = df_merged[mask_keep].copy().reset_index(drop=True)
    n_removed = len(df_merged) - len(df_clean)
    
    print(f'\nĐã lọc: {n_removed} ảnh không hợp lệ')
    print(f'Còn lại: {len(df_clean)} ảnh hợp lệ')
    
    return df_clean

def split_dataset(df_clean: pd.DataFrame) -> pd.DataFrame:
    df_trainval, df_test = train_test_split(
        df_clean,
        test_size=TEST_RATIO,
        stratify=df_clean['label'],
        random_state=SEED
    )
    val_ratio_relative = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_ratio_relative,
        stratify=df_trainval['label'],
        random_state=SEED
    )

    df_train = df_train.copy(); df_train['split'] = 'train'
    df_val   = df_val.copy();   df_val['split']   = 'val'
    df_test  = df_test.copy();  df_test['split']  = 'test'
    df_final = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)

    print('\n── Phân bố sau split ──────────────────────────────────────────')
    summary = df_final.groupby(['split', 'label']).size().unstack(fill_value=0)
    summary['total'] = summary.sum(axis=1)
    summary = summary.loc[['train', 'val', 'test']]
    print(summary.to_string())
    print(f'\nTổng: {len(df_final)}')
    
    return df_final

def safe_copy(src_path: str, dst_path: Path):

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(src_path) as img:
            img_rgb = img.convert('RGB')
            img_rgb.save(str(dst_path), 'JPEG', quality=95)
    except Exception as e:
        print(f'  [SKIP] {src_path}: {e}')
        return False
    return True

def export_data(df_final: pd.DataFrame, root_out: Path):
    """Lưu dữ liệu sang thư mục mới theo format ImageFolder."""
    for split_name in ['train', 'val', 'test']:
        for cls in CLASS_NAMES:
            (root_out / split_name / cls).mkdir(parents=True, exist_ok=True)
            
    print('\nCopy ảnh vào thư mục output...')
    copy_errors = 0

    for _, row in tqdm(df_final.iterrows(), total=len(df_final), desc='Copying'):
        split_name = row['split']
        cls        = row['label']

        stem     = Path(row['filename']).stem
        new_name = f"{row['original_split']}_{stem}.jpg"
        dst      = root_out / split_name / cls / new_name

        if not safe_copy(row['path'], dst):
            copy_errors += 1
        else:
            df_final.loc[row.name, 'new_path'] = str(dst)

    print(f'\nCopy errors: {copy_errors}')
    
    # Save metadata
    csv_path = root_out / 'metadata.csv'
    save_cols = [
        'new_path', 'filename', 'label', 'split',
        'original_split', 'width', 'height', 'mode',
        'file_size_kb', 'is_grayscale', 'md5',
    ]
    df_final[[c for c in save_cols if c in df_final.columns]].to_csv(csv_path, index=False)
    print(f'Saved metadata: {csv_path} ({len(df_final)} rows)')

def main():
    args = parse_args()
    root_in = Path(args.input)
    root_out = Path(args.output)
    
    if not root_in.exists():
        raise FileNotFoundError(f"Thư mục input không tồn tại: {root_in}")
        
    print(f"Input  : {root_in}")
    print(f"Output : {root_out}")
    
    # BƯỚC 1: Đọc dữ liệu
    raw_records = collect_images(root_in)
    df_raw = pd.DataFrame(raw_records)
    print(f'\nTổng ảnh thu thập: {len(df_raw)}')
    
    # BƯỚC 2: Tiền xử lý
    df_clean = preprocess_and_filter(df_raw)
    
    # BƯỚC 3: Chia tập
    df_final = split_dataset(df_clean)
    
    # Xuất Output
    export_data(df_final, root_out)
    
    print("\n✓ Hoàn thành tiền xử lý!")

if __name__ == "__main__":
    main()