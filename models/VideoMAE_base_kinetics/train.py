from __future__ import annotations

import gc
import json
import math
import os
import pickle
import random
import re
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
except ImportError as exc:
    raise ImportError(
        "Missing package: transformers. Install it in Kaggle with: "
        "pip install -q transformers accelerate"
    ) from exc


class CFG:
    DATA_ROOT_HINT = "/kaggle/input/datasets/trngtnbi/signlanguage"
    OUTPUT_ROOT = "/kaggle/working/artifacts/stage5_videomae_fullframe"

    MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"
    LOCAL_MODEL_DIR = ""  # Optional Kaggle dataset path containing a HF model snapshot.

    SEED = 42
    NUM_CLASSES = 100
    NUM_FRAMES = 16
    IMAGE_SIZE = 224
    VAL_SIZE = 0.15

    BATCH_SIZE = 6
    GRAD_ACCUM_STEPS = 4
    NUM_WORKERS = 4
    EPOCHS = 20
    EARLY_STOPPING_PATIENCE = 5

    LR = 2e-5
    WEIGHT_DECAY = 0.05
    WARMUP_RATIO = 0.10
    GRAD_CLIP_NORM = 1.0
    LABEL_SMOOTHING = 0.10
    USE_MIXUP = True
    MIXUP_ALPHA = 0.20

    USE_AMP = True
    USE_WEIGHTED_SAMPLER = True
    USE_DATA_PARALLEL = False
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    TEMPORAL_JITTER = True
    RANDOM_RESIZED_CROP = True
    RANDOM_CROP_SCALE_MIN = 0.90
    COLOR_JITTER = True
    BRIGHTNESS = 0.12
    CONTRAST = 0.12
    SATURATION = 0.08
    HORIZONTAL_FLIP = False  # Do not flip sign-language videos; handedness can change meaning.

    MAX_TRAIN_SAMPLES = None  # Set to a small int for smoke tests.
    MAX_VAL_SAMPLES = None
    MAX_TEST_SAMPLES = None

    SAVE_EVERY_EPOCH = True

    # Fallback only. main() replaces these with VideoMAEImageProcessor stats when available.
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def cfg_to_dict() -> Dict:
    snapshot = {}
    for key, value in vars(CFG).items():
        if key.startswith("_") or callable(value):
            continue
        snapshot[key] = value
    return snapshot


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast_context(enabled: bool):
    if enabled and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def get_script_dir() -> Path | None:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return None


def find_data_root(configured_root: str) -> Path:
    candidates = [
        Path(configured_root),
        Path.cwd() / "data",
    ]
    script_dir = get_script_dir()
    if script_dir is not None:
        candidates.insert(1, script_dir / "data")

    for candidate in candidates:
        if (candidate / "train").exists() and (candidate / "test").exists():
            return candidate

    for base in [Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()]:
        if not base.exists():
            continue
        for mapping_path in base.rglob("label_mapping.pkl"):
            candidate = mapping_path.parent
            if (candidate / "train").exists() and (candidate / "test").exists():
                return candidate

    raise FileNotFoundError(
        "Cannot find dataset root with train/, test/, and label_mapping.pkl. "
        f"Configured root: {configured_root}"
    )


def load_label_mapping(data_root: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    mapping_path = data_root / "label_mapping.pkl"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing label mapping: {mapping_path}")

    with mapping_path.open("rb") as f:
        mapping_obj = pickle.load(f)

    if isinstance(mapping_obj, dict):
        keys = list(mapping_obj.keys())
        values = list(mapping_obj.values())
        if all(isinstance(k, str) for k in keys) and all(isinstance(v, (int, np.integer)) for v in values):
            class_to_id = {str(k): int(v) for k, v in mapping_obj.items()}
            id_to_class = {int(v): str(k) for k, v in class_to_id.items()}
            return class_to_id, id_to_class
        if all(isinstance(k, (int, np.integer)) for k in keys) and all(isinstance(v, str) for v in values):
            id_to_class = {int(k): str(v) for k, v in mapping_obj.items()}
            class_to_id = {str(v): int(k) for k, v in id_to_class.items()}
            return class_to_id, id_to_class

    if isinstance(mapping_obj, (list, tuple)):
        class_to_id = {str(class_name): idx for idx, class_name in enumerate(mapping_obj)}
        id_to_class = {idx: str(class_name) for class_name, idx in class_to_id.items()}
        return class_to_id, id_to_class

    raise ValueError(f"Unsupported label_mapping.pkl format: {type(mapping_obj)}")


def root_id_from_path(path: Path) -> str:
    return re.sub(r"_\d+$", "", path.stem)


def build_train_dataframe(data_root: Path, class_to_id: Dict[str, int]) -> pd.DataFrame:
    rows = []
    train_root = data_root / "train"

    for class_name, label_id in sorted(class_to_id.items(), key=lambda x: x[1]):
        class_dir = train_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        for video_path in sorted(class_dir.glob("*.mp4")):
            root_id = root_id_from_path(video_path)
            rows.append(
                {
                    "video_path": str(video_path),
                    "video_id": video_path.stem,
                    "root_id": root_id,
                    "group_id": f"{label_id}_{root_id}",
                    "class_name": class_name,
                    "label_id": int(label_id),
                }
            )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError(f"No train videos found under {train_root}")
    return df.sort_values(["label_id", "video_id"]).reset_index(drop=True)


def build_test_dataframe(data_root: Path) -> pd.DataFrame:
    test_root = data_root / "test"
    rows = []
    for video_path in sorted(test_root.glob("*.mp4")):
        rows.append(
            {
                "video_path": str(video_path),
                "video_id": video_path.stem,
                "root_id": root_id_from_path(video_path),
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def create_grouped_train_val_split(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    rng = np.random.default_rng(CFG.SEED)
    train_parts = []
    val_parts = []
    warnings = []

    for label_id, label_df in train_df.groupby("label_id", sort=True):
        group_ids = np.array(sorted(label_df["group_id"].unique()))
        rng.shuffle(group_ids)

        if len(group_ids) <= 1:
            train_parts.append(label_df)
            warnings.append(
                {
                    "label_id": int(label_id),
                    "reason": "only_one_group_all_kept_in_train",
                    "num_videos": int(len(label_df)),
                    "num_groups": int(len(group_ids)),
                }
            )
            continue

        n_val_groups = max(1, int(round(len(group_ids) * CFG.VAL_SIZE)))
        n_val_groups = min(n_val_groups, len(group_ids) - 1)
        val_groups = set(group_ids[:n_val_groups].tolist())

        is_val = label_df["group_id"].isin(val_groups)
        train_parts.append(label_df[~is_val])
        val_parts.append(label_df[is_val])

    train_split = pd.concat(train_parts, axis=0).sample(frac=1.0, random_state=CFG.SEED).reset_index(drop=True)
    if val_parts:
        val_split = pd.concat(val_parts, axis=0).sample(frac=1.0, random_state=CFG.SEED).reset_index(drop=True)
    else:
        val_split = pd.DataFrame(columns=train_df.columns)
        warnings.append({"reason": "no_validation_groups_created"})

    train_classes = set(train_split["label_id"].unique().tolist())
    val_classes = set(val_split["label_id"].unique().tolist())
    all_classes = set(train_df["label_id"].unique().tolist())

    report = {
        "split_method": "per_class_grouped_split_by_video_root_id",
        "val_size_target": float(CFG.VAL_SIZE),
        "num_train": int(len(train_split)),
        "num_val": int(len(val_split)),
        "num_total": int(len(train_df)),
        "num_train_classes": int(len(train_classes)),
        "num_val_classes": int(len(val_classes)),
        "missing_train_classes": sorted([int(x) for x in all_classes - train_classes]),
        "missing_val_classes": sorted([int(x) for x in all_classes - val_classes]),
        "warnings": warnings,
    }
    return train_split, val_split, report


def limit_dataframe(df: pd.DataFrame, max_samples: int | None, seed: int) -> pd.DataFrame:
    if max_samples is None or max_samples <= 0 or len(df) <= max_samples:
        return df.reset_index(drop=True)
    return df.sample(n=max_samples, random_state=seed).reset_index(drop=True)


def read_video_frames(video_path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    return frames


def sample_frame_indices(total_frames: int, num_frames: int, train_mode: bool) -> np.ndarray:
    if total_frames <= 0:
        return np.zeros((num_frames,), dtype=np.int64)

    if train_mode and CFG.TEMPORAL_JITTER:
        boundaries = np.linspace(0, total_frames, num_frames + 1)
        indices = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            lo = int(math.floor(start))
            hi = max(lo + 1, int(math.ceil(end)))
            hi = min(hi, total_frames)
            indices.append(random.randrange(lo, hi))
        return np.asarray(indices, dtype=np.int64)

    return np.linspace(0, total_frames - 1, num_frames, dtype=np.int64)


def resize_frame(frame: np.ndarray, size: int) -> np.ndarray:
    if frame.shape[0] == size and frame.shape[1] == size:
        return frame
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)


def random_resized_crop(frame: np.ndarray, size: int) -> np.ndarray:
    if not CFG.RANDOM_RESIZED_CROP:
        return resize_frame(frame, size)

    h, w = frame.shape[:2]
    scale = random.uniform(CFG.RANDOM_CROP_SCALE_MIN, 1.0)
    crop_h = max(1, int(round(h * scale)))
    crop_w = max(1, int(round(w * scale)))

    if crop_h >= h or crop_w >= w:
        return resize_frame(frame, size)

    y1 = random.randint(0, h - crop_h)
    x1 = random.randint(0, w - crop_w)
    crop = frame[y1 : y1 + crop_h, x1 : x1 + crop_w]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


def apply_color_jitter(frames: np.ndarray) -> np.ndarray:
    if not CFG.COLOR_JITTER:
        return frames

    out = frames.astype(np.float32)

    brightness = random.uniform(1.0 - CFG.BRIGHTNESS, 1.0 + CFG.BRIGHTNESS)
    contrast = random.uniform(1.0 - CFG.CONTRAST, 1.0 + CFG.CONTRAST)
    saturation = random.uniform(1.0 - CFG.SATURATION, 1.0 + CFG.SATURATION)

    out = out * brightness
    mean = out.mean(axis=(1, 2), keepdims=True)
    out = (out - mean) * contrast + mean

    gray = np.dot(out[..., :3], np.asarray([0.299, 0.587, 0.114], dtype=np.float32))[..., None]
    out = gray + (out - gray) * saturation

    return np.clip(out, 0, 255).astype(np.uint8)


def video_to_tensor(video_path: str, train_mode: bool) -> torch.Tensor:
    frames = read_video_frames(video_path)

    if len(frames) == 0:
        frames = [np.zeros((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE, 3), dtype=np.uint8)]

    indices = sample_frame_indices(len(frames), CFG.NUM_FRAMES, train_mode=train_mode)
    selected = [frames[int(np.clip(idx, 0, len(frames) - 1))] for idx in indices]

    if train_mode:
        selected = [random_resized_crop(frame, CFG.IMAGE_SIZE) for frame in selected]
        arr = np.stack(selected, axis=0)
        arr = apply_color_jitter(arr)
    else:
        selected = [resize_frame(frame, CFG.IMAGE_SIZE) for frame in selected]
        arr = np.stack(selected, axis=0)

    arr = arr.astype(np.float32) / 255.0
    mean = np.asarray(CFG.MEAN, dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.asarray(CFG.STD, dtype=np.float32).reshape(1, 1, 1, 3)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (0, 3, 1, 2))  # T, C, H, W
    return torch.from_numpy(arr.astype(np.float32))


class VideoClassificationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train_mode: bool, with_labels: bool) -> None:
        self.df = df.reset_index(drop=True)
        self.train_mode = train_mode
        self.with_labels = with_labels

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        pixel_values = video_to_tensor(str(row["video_path"]), train_mode=self.train_mode)
        item = {
            "pixel_values": pixel_values,
            "video_id": str(row["video_id"]),
        }
        if self.with_labels:
            item["labels"] = torch.tensor(int(row["label_id"]), dtype=torch.long)
        return item


def make_train_sampler(train_df: pd.DataFrame) -> WeightedRandomSampler | None:
    if not CFG.USE_WEIGHTED_SAMPLER:
        return None

    counts = train_df["label_id"].value_counts().to_dict()
    weights = train_df["label_id"].map(lambda label: 1.0 / float(counts[int(label)])).to_numpy(dtype=np.float64)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def make_loader(df: pd.DataFrame, train_mode: bool, with_labels: bool) -> DataLoader:
    ds = VideoClassificationDataset(df=df, train_mode=train_mode, with_labels=with_labels)
    sampler = make_train_sampler(df) if train_mode and with_labels else None
    loader_kwargs = {
        "batch_size": CFG.BATCH_SIZE,
        "shuffle": bool(train_mode and sampler is None),
        "sampler": sampler,
        "num_workers": CFG.NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }
    if CFG.NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(ds, **loader_kwargs)


def get_model_source() -> str:
    return CFG.LOCAL_MODEL_DIR.strip() if CFG.LOCAL_MODEL_DIR.strip() else CFG.MODEL_NAME


def configure_image_normalization() -> None:
    model_source = get_model_source()
    try:
        processor = VideoMAEImageProcessor.from_pretrained(model_source)
    except Exception as exc:
        print(f"Could not load VideoMAEImageProcessor. Using fallback MEAN/STD. Reason: {exc}")
        return

    image_mean = getattr(processor, "image_mean", None)
    image_std = getattr(processor, "image_std", None)
    if image_mean is None or image_std is None or len(image_mean) != 3 or len(image_std) != 3:
        print("VideoMAEImageProcessor has no valid image_mean/image_std. Using fallback MEAN/STD.")
        return

    CFG.MEAN = tuple(float(x) for x in image_mean)
    CFG.STD = tuple(float(x) for x in image_std)
    print(f"Using processor normalization | mean={CFG.MEAN} std={CFG.STD}")


def build_model(num_classes: int, id_to_class: Dict[int, str]) -> nn.Module:
    model_source = get_model_source()
    if CFG.LOCAL_MODEL_DIR.strip() and not Path(CFG.LOCAL_MODEL_DIR).exists():
        raise FileNotFoundError(f"LOCAL_MODEL_DIR does not exist: {CFG.LOCAL_MODEL_DIR}")

    label2id = {name: int(idx) for idx, name in id_to_class.items()}
    id2label = {int(idx): name for idx, name in id_to_class.items()}

    model = VideoMAEForVideoClassification.from_pretrained(
        model_source,
        num_labels=num_classes,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    return model


def create_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int):
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_metrics(y_true: List[int], y_pred: List[int], num_classes: int) -> Dict[str, float]:
    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "macro_f1_present": 0.0,
            "weighted_f1": 0.0,
            "unique_pred": 0.0,
        }
    labels = list(range(num_classes))
    present_labels = sorted(set(int(x) for x in y_true))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1_present": float(
            f1_score(y_true, y_pred, labels=present_labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "unique_pred": float(len(set(y_pred))),
    }


def mixup_batch(pixel_values: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if not CFG.USE_MIXUP or CFG.MIXUP_ALPHA <= 0 or pixel_values.size(0) < 2:
        return pixel_values, labels, labels, 1.0

    lam = float(np.random.beta(CFG.MIXUP_ALPHA, CFG.MIXUP_ALPHA))
    indices = torch.randperm(pixel_values.size(0), device=pixel_values.device)
    mixed = lam * pixel_values + (1.0 - lam) * pixel_values[indices]
    return mixed, labels, labels[indices], lam


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler,
    scaler: torch.amp.GradScaler,
    train_mode: bool,
    num_classes: int,
) -> Dict[str, float]:
    model.train(train_mode)
    loss_sum = 0.0
    n_samples = 0
    y_true: List[int] = []
    y_pred: List[int] = []

    if train_mode and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        pixel_values = batch["pixel_values"].to(CFG.DEVICE, non_blocking=True)
        labels = batch["labels"].to(CFG.DEVICE, non_blocking=True)

        with torch.set_grad_enabled(train_mode):
            with autocast_context(enabled=bool(CFG.USE_AMP and torch.cuda.is_available())):
                if train_mode and CFG.USE_MIXUP:
                    model_inputs, labels_a, labels_b, lam = mixup_batch(pixel_values, labels)
                    outputs = model(pixel_values=model_inputs)
                else:
                    labels_a, labels_b, lam = labels, labels, 1.0
                    outputs = model(pixel_values=pixel_values)
                logits = outputs.logits
                loss = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
                scaled_loss = loss / max(1, CFG.GRAD_ACCUM_STEPS)

            if train_mode and optimizer is not None:
                scaler.scale(scaled_loss).backward()
                if step % CFG.GRAD_ACCUM_STEPS == 0:
                    scaler.unscale_(optimizer)
                    if CFG.GRAD_CLIP_NORM is not None and CFG.GRAD_CLIP_NORM > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()

        preds = torch.argmax(logits.detach(), dim=1)
        batch_size = int(labels.shape[0])

        loss_sum += float(loss.detach().item()) * batch_size
        n_samples += batch_size
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    if train_mode and optimizer is not None:
        has_grad = any(p.grad is not None for p in model.parameters())
        if has_grad:
            scaler.unscale_(optimizer)
            if CFG.GRAD_CLIP_NORM is not None and CFG.GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

    metrics = compute_metrics(y_true, y_pred, num_classes=num_classes)
    metrics["loss"] = float(loss_sum / max(1, n_samples))
    metrics["samples"] = int(n_samples)
    return metrics


@torch.no_grad()
def predict_dataframe(model: nn.Module, df: pd.DataFrame, with_labels: bool) -> Dict[str, np.ndarray]:
    loader = make_loader(df, train_mode=False, with_labels=with_labels)
    model.eval()

    probs_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    video_ids_all: List[str] = []

    for batch in loader:
        pixel_values = batch["pixel_values"].to(CFG.DEVICE, non_blocking=True)
        with autocast_context(enabled=bool(CFG.USE_AMP and torch.cuda.is_available())):
            logits = model(pixel_values=pixel_values).logits
            probs = torch.softmax(logits, dim=1)

        probs_all.append(probs.float().cpu().numpy())
        video_ids_all.extend([str(x) for x in batch["video_id"]])
        if with_labels:
            labels_all.append(batch["labels"].cpu().numpy())

    result = {
        "probs": np.concatenate(probs_all, axis=0) if probs_all else np.zeros((0, 0), dtype=np.float32),
        "video_ids": np.asarray(video_ids_all, dtype=object),
    }
    if with_labels:
        result["labels"] = np.concatenate(labels_all, axis=0) if labels_all else np.asarray([], dtype=np.int64)
    return result


def save_checkpoint(model: nn.Module, path: Path, epoch: int, best_val_f1: float, cfg_snapshot: Dict) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": unwrap_model(model).state_dict(),
            "best_val_macro_f1": float(best_val_f1),
            "cfg": cfg_snapshot,
        },
        path,
    )


def load_checkpoint(path: Path, map_location: str):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def main() -> None:
    set_seed(CFG.SEED)
    print(f"Device: {CFG.DEVICE}")
    print(f"CUDA devices: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")

    data_root = find_data_root(CFG.DATA_ROOT_HINT)
    output_root = Path(CFG.OUTPUT_ROOT)
    checkpoints_dir = output_root / "checkpoints"
    logs_dir = output_root / "logs"
    metadata_dir = output_root / "metadata"
    configs_dir = output_root / "configs"

    for path in [checkpoints_dir, logs_dir, metadata_dir, configs_dir]:
        ensure_dir(path)

    class_to_id, id_to_class = load_label_mapping(data_root)
    num_classes = len(class_to_id)
    if num_classes != CFG.NUM_CLASSES:
        print(f"Warning: CFG.NUM_CLASSES={CFG.NUM_CLASSES}, label mapping has {num_classes}. Using mapping count.")
    configure_image_normalization()

    train_all_df = build_train_dataframe(data_root, class_to_id)
    test_df = build_test_dataframe(data_root)
    train_df, val_df, split_report = create_grouped_train_val_split(train_all_df)

    train_df = limit_dataframe(train_df, CFG.MAX_TRAIN_SAMPLES, CFG.SEED)
    val_df = limit_dataframe(val_df, CFG.MAX_VAL_SAMPLES, CFG.SEED)
    test_df = limit_dataframe(test_df, CFG.MAX_TEST_SAMPLES, CFG.SEED)

    class_distribution = (
        train_all_df.groupby(["label_id", "class_name"], as_index=False)
        .agg(num_videos=("video_path", "count"), num_groups=("group_id", "nunique"))
        .sort_values("label_id")
    )

    train_all_df.to_csv(metadata_dir / "train_all_metadata.csv", index=False, encoding="utf-8-sig")
    train_df.to_csv(metadata_dir / "train_split.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(metadata_dir / "val_split.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(metadata_dir / "test_metadata.csv", index=False, encoding="utf-8-sig")
    class_distribution.to_csv(metadata_dir / "class_distribution.csv", index=False, encoding="utf-8-sig")

    cfg_snapshot = cfg_to_dict()
    save_json({"cfg": cfg_snapshot}, configs_dir / "stage5_config.json")
    save_json(split_report, configs_dir / "grouped_split_report.json")
    save_json({str(k): v for k, v in id_to_class.items()}, configs_dir / "id_to_class.json")
    save_json({k: int(v) for k, v in class_to_id.items()}, configs_dir / "class_to_id.json")

    print(f"Data root: {data_root}")
    print(f"Output root: {output_root}")
    print(f"Model: {CFG.LOCAL_MODEL_DIR if CFG.LOCAL_MODEL_DIR else CFG.MODEL_NAME}")
    print(f"Classes: {num_classes}")
    print(f"Train all videos: {len(train_all_df)}")
    print(f"Grouped train videos: {len(train_df)}")
    print(f"Grouped val videos: {len(val_df)}")
    print(f"Test videos: {len(test_df)}")
    print(
        f"Train cfg | frames={CFG.NUM_FRAMES} image={CFG.IMAGE_SIZE} batch={CFG.BATCH_SIZE} "
        f"accum={CFG.GRAD_ACCUM_STEPS} epochs={CFG.EPOCHS} lr={CFG.LR} "
        f"label_smoothing={CFG.LABEL_SMOOTHING} mixup={CFG.USE_MIXUP} "
        f"mixup_alpha={CFG.MIXUP_ALPHA} weighted_sampler={CFG.USE_WEIGHTED_SAMPLER}"
    )
    if split_report["missing_train_classes"] or split_report["missing_val_classes"]:
        print(f"Missing train classes: {split_report['missing_train_classes']}")
        print(f"Missing val classes: {split_report['missing_val_classes']}")

    train_loader = make_loader(train_df, train_mode=True, with_labels=True)
    val_loader = make_loader(val_df, train_mode=False, with_labels=True)

    model = build_model(num_classes=num_classes, id_to_class=id_to_class).to(CFG.DEVICE)
    if CFG.USE_DATA_PARALLEL and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs.")

    criterion = nn.CrossEntropyLoss(label_smoothing=CFG.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)

    optimizer_steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, CFG.GRAD_ACCUM_STEPS)))
    total_optimizer_steps = optimizer_steps_per_epoch * CFG.EPOCHS
    warmup_steps = int(round(total_optimizer_steps * CFG.WARMUP_RATIO))
    scheduler = create_scheduler(optimizer, total_steps=total_optimizer_steps, warmup_steps=warmup_steps)
    scaler = torch.amp.GradScaler(enabled=bool(CFG.USE_AMP and torch.cuda.is_available()))

    best_val_f1 = -1.0
    best_epoch = -1
    patience_left = CFG.EARLY_STOPPING_PATIENCE
    history_rows = []
    start_time = time.time()

    best_path = checkpoints_dir / "best_videomae_fullframe.pth"
    last_path = checkpoints_dir / "last_videomae_fullframe.pth"

    for epoch in range(1, CFG.EPOCHS + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_mode=True,
            num_classes=num_classes,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            scheduler=None,
            scaler=scaler,
            train_mode=False,
            num_classes=num_classes,
        )

        lr_now = float(optimizer.param_groups[0]["lr"])
        is_best = val_metrics["macro_f1"] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            patience_left = CFG.EARLY_STOPPING_PATIENCE
            save_checkpoint(model, best_path, epoch, best_val_f1, cfg_snapshot)
            unwrap_model(model).save_pretrained(checkpoints_dir / "best_hf_model")
        else:
            patience_left -= 1

        if CFG.SAVE_EVERY_EPOCH:
            save_checkpoint(model, last_path, epoch, best_val_f1, cfg_snapshot)

        epoch_row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_macro_f1_present": train_metrics["macro_f1_present"],
            "train_weighted_f1": train_metrics["weighted_f1"],
            "train_unique_pred": train_metrics["unique_pred"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_macro_f1_present": val_metrics["macro_f1_present"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_unique_pred": val_metrics["unique_pred"],
            "best_val_macro_f1": best_val_f1,
            "best_epoch": best_epoch,
            "is_best": int(is_best),
            "patience_left": patience_left,
            "elapsed_minutes": round((time.time() - start_time) / 60.0, 3),
        }
        history_rows.append(epoch_row)
        pd.DataFrame(history_rows).to_csv(logs_dir / "training_log.csv", index=False, encoding="utf-8-sig")

        print(
            f"Epoch {epoch}/{CFG.EPOCHS} | "
            f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['macro_f1']:.5f} "
            f"train_f1_present={train_metrics['macro_f1_present']:.5f} "
            f"train_acc={train_metrics['accuracy']:.4f} pred_cls={int(train_metrics['unique_pred'])} | "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['macro_f1']:.5f} "
            f"val_f1_present={val_metrics['macro_f1_present']:.5f} "
            f"val_acc={val_metrics['accuracy']:.4f} pred_cls={int(val_metrics['unique_pred'])} | "
            f"best={best_val_f1:.5f} patience={patience_left}",
            flush=True,
        )

        if patience_left <= 0:
            print("Early stopping triggered.")
            break

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best_ckpt = load_checkpoint(best_path, map_location=CFG.DEVICE)
    unwrap_model(model).load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    val_pred = predict_dataframe(model, val_df, with_labels=True)
    val_top1 = np.argmax(val_pred["probs"], axis=1) if val_pred["probs"].size else np.asarray([], dtype=np.int64)
    val_pred_df = pd.DataFrame(
        {
            "video_id": val_pred["video_ids"].astype(str),
            "label_id": val_pred["labels"].astype(np.int64),
            "pred_label_id": val_top1.astype(np.int64),
            "pred_class": [id_to_class.get(int(x), "") for x in val_top1],
        }
    )
    np.save(metadata_dir / "val_videomae_probs.npy", val_pred["probs"])
    np.save(metadata_dir / "val_videomae_labels.npy", val_pred["labels"])
    np.save(metadata_dir / "val_videomae_video_ids.npy", val_pred["video_ids"])
    val_pred_df.to_csv(metadata_dir / "val_videomae_predictions.csv", index=False, encoding="utf-8-sig")

    test_summary = {}
    if len(test_df) > 0:
        test_pred = predict_dataframe(model, test_df, with_labels=False)
        test_top1 = np.argmax(test_pred["probs"], axis=1) if test_pred["probs"].size else np.asarray([], dtype=np.int64)
        test_pred_df = pd.DataFrame(
            {
                "video_id": test_pred["video_ids"].astype(str),
                "pred_label_id": test_top1.astype(np.int64),
                "pred_class": [id_to_class.get(int(x), "") for x in test_top1],
            }
        )
        np.save(metadata_dir / "test_videomae_probs.npy", test_pred["probs"])
        np.save(metadata_dir / "test_videomae_video_ids.npy", test_pred["video_ids"])
        test_pred_df.to_csv(metadata_dir / "test_videomae_predictions.csv", index=False, encoding="utf-8-sig")
        test_pred_df[["video_id", "pred_label_id"]].to_csv(
            output_root / "submission.csv",
            index=False,
            encoding="utf-8-sig",
        )
        test_summary = {
            "test_probs_path": str(metadata_dir / "test_videomae_probs.npy"),
            "test_video_ids_path": str(metadata_dir / "test_videomae_video_ids.npy"),
            "test_predictions_csv": str(metadata_dir / "test_videomae_predictions.csv"),
            "submission_csv": str(output_root / "submission.csv"),
        }

    summary = {
        "data_root": str(data_root),
        "output_root": str(output_root),
        "model_name": CFG.LOCAL_MODEL_DIR if CFG.LOCAL_MODEL_DIR else CFG.MODEL_NAME,
        "num_classes": int(num_classes),
        "train_all_videos": int(len(train_all_df)),
        "train_videos": int(len(train_df)),
        "val_videos": int(len(val_df)),
        "test_videos": int(len(test_df)),
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_f1),
        "best_checkpoint": str(best_path),
        "val_probs_path": str(metadata_dir / "val_videomae_probs.npy"),
        "val_labels_path": str(metadata_dir / "val_videomae_labels.npy"),
        "val_video_ids_path": str(metadata_dir / "val_videomae_video_ids.npy"),
        "val_predictions_csv": str(metadata_dir / "val_videomae_predictions.csv"),
        **test_summary,
    }
    save_json(summary, configs_dir / "stage5_training_summary.json")

    print("\nStage 5 done.")
    print(f"Best checkpoint: {best_path}")
    print(f"Best val macro F1: {best_val_f1:.6f} at epoch {best_epoch}")
    print(f"Summary: {configs_dir / 'stage5_training_summary.json'}")


if __name__ == "__main__":
    main()
