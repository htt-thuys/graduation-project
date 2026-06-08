# Ứng dụng Supervised Contrastive Learning (SupCon) cho Phân Loại Viêm Phổi qua Ảnh X-Quang Ngực

Đây là kho lưu trữ chứa toàn bộ mã nguồn cho Khóa luận tốt nghiệp của tôi. Dự án này tập trung vào việc áp dụng phương pháp học đối chiếu có giám sát (**Supervised Contrastive Learning - SupCon**) để phát hiện và phân loại bệnh viêm phổi (Pneumonia) từ hình ảnh X-quang ngực (Chest X-Ray).

---

## Dataset Sử Dụng
Dự án sử dụng bộ dữ liệu **[Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)** từ Kaggle.

**Đặc điểm của bộ dữ liệu sau khi tiền xử lý:**
- Tổng số lượng ảnh: **5,824** ảnh X-quang ngực.
- Các nhãn (Classes):
  - `NORMAL` (Bình thường): 1,579 ảnh.
  - `PNEUMONIA` (Viêm phổi): 4,245 ảnh.
- Phân chia tập dữ liệu (Stratified Split): **70% Train / 15% Validation / 15% Test** nhằm giải quyết vấn đề chia tập mất cân bằng ở dữ liệu gốc của Kaggle và để đảm bảo quá trình huấn luyện/đánh giá khách quan.

---

## Cấu Trúc Thư Mục Codebase
Trong phiên bản codebase mới này, toàn bộ quá trình nghiên cứu trên Jupyter Notebook (`.ipynb`) đã được tái cấu trúc, module hóa thành các file Python (`.py`) độc lập. Điều này giúp mã nguồn trở nên gọn gàng, dễ dàng mở rộng và triển khai (deployment).

```text
codebase/
├── configs/                # Cấu hình dự án
├── outputs/                # Nơi lưu kết quả
│   ├── checkpoints/        # Model weights (.pth)
│   ├── figures/            # Đồ thị, Heatmap, t-SNE
│   └── logs/               # TensorBoard logs, file text log
├── scripts/                # Các script phụ trợ
│   └── test_local.py       # Local testing script
├── src/                    # Mã nguồn chính (được import lại)
│   ├── data/
│   │   ├── preprocess.py   # Code tiền xử lý
│   │   └── transforms.py   # Các logic augmentation
│   ├── models/
│   │   ├── backbone.py     # Định nghĩa kiến trúc mô hình
│   │   └── losses.py       # Loss function
│   └── utils/              # Các hàm dùng chung
└── tools/                  # Script chạy chính 
    ├── train_stage1.py
    ├── train_stage2.py
    └── evaluate.py
```

---

## Các Giai Đoạn / Tác Vụ Của Dự Án

### 1. Tiền xử lý dữ liệu (`preprocess.py`)
Mục đích: Khắc phục các vấn đề của dataset gốc như tập validation quá nhỏ (chỉ 16 ảnh), ảnh hỏng, và kích thước bất thường.
- Đọc, kiểm tra và lọc các ảnh lỗi/trùng lặp bằng MD5.
- Thực hiện **Stratified Split (70/15/15)** để chia lại tập Train/Val/Test.
- Lưu trữ meta-data vào `metadata.csv`.

### 2. Huấn luyện Stage 1: SupCon Pre-training (`train_stage1.py`)
Mục đích: Huấn luyện mạng tạo đặc trưng (Encoder) học cách kéo các đặc trưng của cùng một lớp lại gần nhau và đẩy các lớp khác biệt ra xa trong không gian nhúng (embedding space).
- Sử dụng hàm mất mát `SupConLoss`.
- Áp dụng các kỹ thuật augmentation mạnh mẽ như RandomResizedCrop, ColorJitter.

### 3. Huấn luyện Stage 2: Classifier Fine-tuning (`train_stage2.py`)
Mục đích: Đóng băng (freeze) mạng encoder (đã được pre-trained ở Stage 1) và chỉ huấn luyện một lớp phân loại tuyến tính (Linear Classifier) trên cùng.
- Sử dụng hàm mất mát Cross Entropy hoặc Focal Loss để tối ưu.
- Hỗ trợ unfreeze encoder ở các epoch sau để fine-tuning toàn bộ mô hình (nếu cần).

### 4. Đánh giá mô hình (`evaluate.py`)
Mục đích: Đo lường hiệu năng của mô hình trên tập Test độc lập.
- Tính toán các chỉ số: **Accuracy, AUC, F1-macro, Recall**.
- So sánh hiệu năng của mô hình SupCon đề xuất với các phương pháp Baseline truyền thống.

### 5. Trực quan hóa (`visualization.py`)
Mục đích: Hiểu rõ những gì mô hình đã học được và tính minh bạch của kết quả.
- **t-SNE / UMAP:** Trực quan hóa sự phân tách các cụm đặc trưng (embeddings) trong không gian 2D/3D.
- **Confusion Matrix:** Đánh giá số lượng dự đoán đúng/sai trên từng nhãn.
- **Grad-CAM / Saliency Map:** Xem vùng ảnh nào trên phim X-quang được mô hình tập trung vào để đưa ra quyết định dự đoán viêm phổi.

---

## Hướng Dẫn Cài Đặt và Chạy Project

### Cài đặt môi trường
Đảm bảo bạn có môi trường Python 3.8+. Kích hoạt môi trường ảo và cài đặt thư viện:

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

### Luồng thực thi mẫu (Execution Flow)
Chạy lần lượt các script theo thứ tự sau để tái lập lại toàn bộ quy trình. Đảm bảo bạn đứng ở thư mục gốc `codebase/`:

```bash
# 1. Tiền xử lý dữ liệu
python src/data/preprocess.py --input /path/to/raw_dataset --output /path/to/processed_dataset

# 2. Stage 1 - SupCon Pretraining (VD: dùng DenseNet121)
python tools/train_stage1.py --backbone densenet121 --train-dir /path/to/processed_dataset/train --out-dir outputs/stage1

# 3. Stage 2 - Classifier Fine-tuning
python tools/train_stage2.py --backbone densenet121 --s1-dir outputs/stage1 --train-dir /path/to/processed_dataset/train --val-dir /path/to/processed_dataset/val --out-dir outputs/stage2

# 4. Đánh giá trên tập test
python tools/evaluate.py --backbone densenet121 --s2-dir outputs/stage2 --test-dir /path/to/processed_dataset/test --out-dir outputs/eval

# 5. Chạy Local Test (Để xác nhận codebase hoạt động tốt)
python scripts/test_local.py
```
