# Sign Language

Demo nhận diện Vietnamese isolated sign language từ video `.mp4`.

Project gồm 2 phần chính:

- Visualize video với bounding box/landmark overlay cho `face`, `body`, `left hand`, `right hand`.
- Predict nhãn tiếng Việt bằng VideoMAE fine-tuned model.

Bounding box chỉ dùng để minh họa trực quan. Model prediction vẫn dùng full-frame video.

## Demo UI

Giao diện Gradio có:

- Input: upload một video `.mp4`.
- Output: video đã overlay landmark/bounding box.
- Output: predicted label tiếng Việt và confidence.

File demo chính:

```txt
frontend/app.py
```

Chạy nhanh trên Windows:

```bat
run_demo.bat
```

Hoặc chạy bằng Python:

```bash
python frontend/app.py
```

Sau đó mở:

```txt
http://127.0.0.1:7860
```

## Clone project

```bash
git clone https://github.com/24521573/SignLanguage.git
cd SignLanguage
```

## Cài đặt môi trường

Khuyến nghị dùng virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Windows CMD:

```bat
.\.venv\Scripts\activate.bat
```

Cài các package cần thiết:

```bash
pip install gradio opencv-python imageio-ffmpeg numpy torch transformers mediapipe
```

## Model artifacts

Các checkpoint/model weight lớn không được push trực tiếp lên GitHub.

Demo hiện dùng Base VideoMAE artifact ở path:

```txt
models/VideoMAE_base_kinetics/artifacts/stage5_videomae_fullframe/
```

Cần có tối thiểu các file sau:

```txt
models/VideoMAE_base_kinetics/artifacts/stage5_videomae_fullframe/
├── checkpoints/
│   └── best_hf_model/
│       ├── config.json
│       └── model.safetensors
└── configs/
    └── id_to_class.json
```

Nếu thiếu các file trên, UI vẫn có thể mở nhưng prediction sẽ không chạy được.

MediaPipe landmark models dùng cho bounding box đã được đặt trong repo:

```txt
models/mediapipe_tasks/
├── face_landmarker.task
├── hand_landmarker.task
└── pose_landmarker_lite.task
```

## Cấu trúc thư mục

```txt
SignLanguage/
├── frontend/
│   └── app.py
├── models/
│   ├── VideoMAE_base_kinetics/
│   │   └── train.py
│   ├── VideoMAE_large_kinetics/
│   │   └── train.py
│   └── mediapipe_tasks/
│       ├── face_landmarker.task
│       ├── hand_landmarker.task
│       └── pose_landmarker_lite.task
├── run_demo.bat
├── .gitignore
└── README.md
```

Các folder không nên push trực tiếp:

```txt
data/
outputs/
models/**/artifacts/
```

## Training scripts

Repo có lưu training scripts:

```txt
models/VideoMAE_base_kinetics/train.py
models/VideoMAE_large_kinetics/train.py
```

Trong demo hiện tại, inference dùng Base VideoMAE artifact để ưu tiên chạy gọn và ổn định.

## Ghi chú

- Input demo chỉ hỗ trợ `.mp4`.
- Video được sample 16 frame.
- Kích thước input model: `224 x 224`.
- Normalization dùng ImageNet mean/std:

```txt
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

