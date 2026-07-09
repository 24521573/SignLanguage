# Sign Language

Gradio demo for Vietnamese isolated sign language recognition from `.mp4` videos.

The project has two main parts:

- Visualize the uploaded video with landmark/bounding-box overlays for `face`, `body`, `left hand`, and `right hand`.
- Predict the Vietnamese sign label using a fine-tuned VideoMAE model.

The bounding boxes are used for visualization only. The prediction model uses the full-frame video as input.

## Demo UI

The Gradio interface provides:

- An `.mp4` video upload input.
- A visualized video output with MediaPipe landmark overlays.
- A prediction card showing the Vietnamese label and confidence score.

Main demo file:

```txt
frontend/app.py
```

Run on Windows:

```bat
run_demo.bat
```

Or run directly with Python:

```bash
python frontend/app.py
```

Then open:

```txt
http://127.0.0.1:7860
```

## Clone the repository

```bash
git clone https://github.com/24521573/SignLanguage.git
cd SignLanguage
```

## Environment setup

Using a virtual environment is recommended:

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

Install the required packages:

```bash
pip install gradio opencv-python imageio-ffmpeg numpy torch transformers mediapipe
```

## Model artifacts

Large checkpoints and model weights are not committed to this repository.

The demo expects the Base VideoMAE artifact at:

```txt
models/VideoMAE_base_kinetics/artifacts/stage5_videomae_fullframe/
```

The minimum required files are:

```txt
models/VideoMAE_base_kinetics/artifacts/stage5_videomae_fullframe/
├── checkpoints/
│   └── best_hf_model/
│       ├── config.json
│       └── model.safetensors
└── configs/
    └── id_to_class.json
```

If these files are missing, the UI can still launch, but prediction will not work.

MediaPipe landmark models for visualization are included in the repository:

```txt
models/mediapipe_tasks/
├── face_landmarker.task
├── hand_landmarker.task
└── pose_landmarker_lite.task
```

## Project structure

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

The following folders are intentionally ignored and should not be pushed directly:

```txt
data/
outputs/
models/**/artifacts/
```

## Training scripts

Training scripts are included for reference:

```txt
models/VideoMAE_base_kinetics/train.py
models/VideoMAE_large_kinetics/train.py
```

The current demo uses the Base VideoMAE artifact for a simpler and more stable local demo.

## Notes

- The demo input supports `.mp4` files only.
- Each video is sampled to 16 frames for model inference.
- Model input size is `224 x 224`.
- Normalization uses ImageNet mean/std:

```txt
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

