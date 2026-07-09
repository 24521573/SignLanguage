from __future__ import annotations

import json
import logging
import threading
import uuid
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import imageio_ffmpeg
import numpy as np
import torch
from transformers import VideoMAEForVideoClassification


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
BASE_ARTIFACT_DIR = (
    PROJECT_ROOT
    / "models"
    / "VideoMAE_base_kinetics"
    / "artifacts"
    / "stage5_videomae_fullframe"
)
HF_MODEL_DIR = BASE_ARTIFACT_DIR / "checkpoints" / "best_hf_model"
LABEL_MAP_PATH = BASE_ARTIFACT_DIR / "configs" / "id_to_class.json"
MEDIAPIPE_TASK_DIR = PROJECT_ROOT / "models" / "mediapipe_tasks"
HAND_TASK_PATH = MEDIAPIPE_TASK_DIR / "hand_landmarker.task"
POSE_TASK_PATH = MEDIAPIPE_TASK_DIR / "pose_landmarker_lite.task"
FACE_TASK_PATH = MEDIAPIPE_TASK_DIR / "face_landmarker.task"

NUM_FRAMES = 16
IMAGE_SIZE = 224
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)
SUPPORTED_EXTENSIONS = {".mp4"}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("sign-language-demo")
FACE_CASCADE = cv2.CascadeClassifier(
    str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
)


class LazyVideoMAEPredictor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: VideoMAEForVideoClassification | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._id_to_class: dict[int, str] | None = None

    @property
    def id_to_class(self) -> dict[int, str]:
        if self._id_to_class is None:
            if not LABEL_MAP_PATH.exists():
                raise FileNotFoundError(f"Không tìm thấy label map: {LABEL_MAP_PATH}")
            raw = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
            self._id_to_class = {int(key): str(value) for key, value in raw.items()}
        return self._id_to_class

    @property
    def model(self) -> VideoMAEForVideoClassification:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._load_model()
        return self._model

    def _load_model(self) -> VideoMAEForVideoClassification:
        if not HF_MODEL_DIR.exists():
            raise FileNotFoundError(f"Không tìm thấy model artifact: {HF_MODEL_DIR}")

        id_to_class = self.id_to_class
        label2id = {label: idx for idx, label in id_to_class.items()}

        LOGGER.info("Loading VideoMAE model from %s", HF_MODEL_DIR)
        model = VideoMAEForVideoClassification.from_pretrained(
            str(HF_MODEL_DIR),
            num_labels=len(id_to_class),
            id2label=id_to_class,
            label2id=label2id,
        )
        model.to(self._device)
        model.eval()
        LOGGER.info("VideoMAE loaded on %s", self._device)
        return model

    @torch.inference_mode()
    def predict(self, video_path: str | Path) -> tuple[str, float]:
        pixel_values = preprocess_video_for_videomae(video_path).to(self._device)
        logits = self.model(pixel_values=pixel_values).logits
        probabilities = torch.softmax(logits, dim=-1)[0]
        confidence, class_id = torch.max(probabilities, dim=-1)
        label = self.id_to_class[int(class_id.item())]
        return label, float(confidence.item() * 100.0)


PREDICTOR = LazyVideoMAEPredictor()


def preprocess_video_for_videomae(video_path: str | Path) -> torch.Tensor:
    frames, _fps = read_video_frames(video_path)
    if not frames:
        frames = [np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)]

    indices = np.linspace(0, len(frames) - 1, NUM_FRAMES, dtype=np.int64)
    selected = [
        cv2.resize(frames[int(idx)], (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        for idx in indices
    ]
    arr = np.stack(selected, axis=0).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (0, 3, 1, 2))
    return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)


def read_video_frames(video_path: str | Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Không mở được video. Hãy thử file .mp4 khác.")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise ValueError("Video không có frame hợp lệ.")
    return frames, max(fps, 1.0)


def create_bbox_visualization(video_path: str | Path) -> str:
    frames_rgb, fps = read_video_frames(video_path)
    frames_rgb = trim_leading_static_frames(frames_rgb)
    try:
        if mediapipe_task_models_ready():
            frames_bgr = draw_mediapipe_task_boxes(frames_rgb, fps)
        else:
            frames_bgr = draw_fallback_boxes(frames_rgb)
    except Exception as error:
        LOGGER.exception("MediaPipe task visualization failed: %s", error)
        frames_bgr = draw_fallback_boxes(frames_rgb)

    output_path = OUTPUT_DIR / f"sign_language_visual_{uuid.uuid4().hex[:10]}.mp4"
    write_browser_mp4(frames_bgr, fps=fps, output_path=output_path)
    return str(output_path)


def mediapipe_task_models_ready() -> bool:
    return HAND_TASK_PATH.exists() and POSE_TASK_PATH.exists() and FACE_TASK_PATH.exists()


def trim_leading_static_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    if len(frames) <= 1:
        return frames

    last_allowed_start = max(0, int(len(frames) * 0.9) - 1)
    start = 0
    for index, frame in enumerate(frames[: last_allowed_start + 1]):
        spatial_std = float(np.mean(frame.std(axis=(0, 1))))
        if spatial_std > 2.0:
            start = index
            break
    return frames[start:]


def draw_mediapipe_task_boxes(frames_rgb: list[np.ndarray], fps: float) -> list[np.ndarray]:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    running_mode = vision.RunningMode.VIDEO
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(HAND_TASK_PATH)),
        running_mode=running_mode,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(POSE_TASK_PATH)),
        running_mode=running_mode,
        min_pose_detection_confidence=0.35,
        min_pose_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    face_options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(FACE_TASK_PATH)),
        running_mode=running_mode,
        num_faces=1,
        min_face_detection_confidence=0.35,
        min_face_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )

    output: list[np.ndarray] = []
    timestamp_step_ms = max(1, int(round(1000.0 / max(float(fps), 1.0))))
    with (
        vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker,
        vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker,
        vision.FaceLandmarker.create_from_options(face_options) as face_landmarker,
    ):
        for index, frame_rgb in enumerate(frames_rgb):
            timestamp_ms = index * timestamp_step_ms
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(frame_rgb),
            )
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            h, w = frame_bgr.shape[:2]

            pose_box = task_landmarks_box(
                pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None,
                w,
                h,
                padding=0.08,
                use_visibility=True,
            )
            if pose_box is not None:
                draw_box(frame_bgr, pose_box, "body", (80, 220, 120))

            face_box = task_landmarks_box(
                face_result.face_landmarks[0] if face_result.face_landmarks else None,
                w,
                h,
                padding=0.10,
            )
            if face_box is not None:
                draw_box(frame_bgr, face_box, "face", (80, 220, 255))

            for label, box in task_hand_boxes(hand_result, w, h).items():
                color = (255, 120, 80) if label == "left hand" else (80, 180, 255)
                draw_box(frame_bgr, box, label, color)

            output.append(frame_bgr)

    return output


def task_hand_boxes(hand_result: Any, width: int, height: int) -> dict[str, tuple[int, int, int, int]]:
    landmarks_list = list(getattr(hand_result, "hand_landmarks", []) or [])
    handedness_list = list(getattr(hand_result, "handedness", []) or [])
    if not landmarks_list:
        return {}

    detected: list[tuple[str, tuple[int, int, int, int]]] = []
    for index, landmarks in enumerate(landmarks_list):
        box = task_landmarks_box(landmarks, width, height, padding=0.16)
        if box is None:
            continue

        label = ""
        if index < len(handedness_list) and handedness_list[index]:
            category = handedness_list[index][0]
            label = str(getattr(category, "category_name", "")).strip().lower()
        if label not in {"left", "right"}:
            label = "left" if box_center(box)[0] < width / 2.0 else "right"
        detected.append((label, box))

    if not detected:
        return {}
    if len(detected) == 1:
        label, box = detected[0]
        return {f"{label} hand": box}

    detected = sorted(detected[:2], key=lambda item: box_center(item[1])[0])
    return {
        "left hand": detected[0][1],
        "right hand": detected[1][1],
    }


def task_landmarks_box(
    landmarks: Any,
    width: int,
    height: int,
    padding: float = 0.10,
    use_visibility: bool = False,
) -> tuple[int, int, int, int] | None:
    if landmarks is None:
        return None

    points: list[tuple[int, int]] = []
    for landmark in landmarks:
        if use_visibility:
            visibility = float(getattr(landmark, "visibility", 1.0) or 0.0)
            presence = float(getattr(landmark, "presence", 1.0) or 0.0)
            if visibility < 0.20 or presence < 0.20:
                continue

        x = float(getattr(landmark, "x", -1.0))
        y = float(getattr(landmark, "y", -1.0))
        if -0.15 <= x <= 1.15 and -0.15 <= y <= 1.15:
            points.append(
                (
                    int(np.clip(x, 0.0, 1.0) * (width - 1)),
                    int(np.clip(y, 0.0, 1.0) * (height - 1)),
                )
            )

    if not points:
        return None

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return expand_box((min(xs), min(ys), max(xs), max(ys)), padding, width, height)


def draw_fallback_boxes(frames_rgb: list[np.ndarray]) -> list[np.ndarray]:
    output: list[np.ndarray] = []

    for index, frame_rgb in enumerate(frames_rgb):
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        detected_face_box = detect_face_box(frame_rgb)
        body_box = estimate_body_box(frame_rgb, detected_face_box)
        face_box = detected_face_box or estimate_face_from_body(body_box, frame_rgb.shape[1], frame_rgb.shape[0])
        previous_rgb = frames_rgb[index - 1] if index > 0 else None
        next_rgb = frames_rgb[index + 1] if index + 1 < len(frames_rgb) else None
        hand_boxes = detect_hand_boxes(
            frame_rgb,
            previous_rgb,
            next_rgb,
            face_box,
            body_box,
            face_is_estimated=detected_face_box is None,
        )

        if body_box is not None:
            draw_box(frame_bgr, body_box, "body", (80, 220, 120))
        if face_box is not None:
            draw_box(frame_bgr, face_box, "face", (80, 220, 255))
        if "left hand" in hand_boxes:
            draw_box(frame_bgr, hand_boxes["left hand"], "left hand", (255, 120, 80))
        if "right hand" in hand_boxes:
            draw_box(frame_bgr, hand_boxes["right hand"], "right hand", (80, 180, 255))

        output.append(frame_bgr)

    return output


def detect_face_box(frame_rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    if FACE_CASCADE.empty():
        return None

    h, w = frame_rgb.shape[:2]
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    min_size = max(18, int(min(h, w) * 0.12))
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(min_size, min_size),
    )
    if len(faces) == 0:
        return None

    x, y, bw, bh = max(faces, key=lambda item: item[2] * item[3])
    return expand_box((x, y, x + bw, y + bh), 0.12, w, h)


def estimate_body_box(
    frame_rgb: np.ndarray,
    face_box: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    h, w = frame_rgb.shape[:2]
    detected = foreground_bbox(frame_rgb)
    if detected is not None:
        area = box_area(detected)
        if h * w * 0.03 <= area <= h * w * 0.72:
            return detected

    if face_box is not None:
        fx1, fy1, fx2, fy2 = face_box
        face_w = max(1, fx2 - fx1)
        face_h = max(1, fy2 - fy1)
        center_x = (fx1 + fx2) // 2
        body_w = int(face_w * 4.4)
        return clamp_box(
            (
                center_x - body_w // 2,
                max(0, fy1 - int(face_h * 0.25)),
                center_x + body_w // 2,
                min(h - 1, fy2 + int(face_h * 4.6)),
            ),
            w,
            h,
        )

    return (
        int(w * 0.18),
        int(h * 0.06),
        int(w * 0.82),
        int(h * 0.96),
    )


def estimate_face_from_body(
    body_box: tuple[int, int, int, int] | None,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    if body_box is None:
        return None

    x1, y1, x2, y2 = body_box
    body_w = max(1, x2 - x1)
    body_h = max(1, y2 - y1)
    face_w = int(body_w * 0.28)
    face_h = int(body_h * 0.24)
    center_x = x1 + int(body_w * 0.52)
    top_y = y1 + int(body_h * 0.08)
    return clamp_box(
        (
            center_x - face_w // 2,
            top_y,
            center_x + face_w // 2,
            top_y + face_h,
        ),
        width,
        height,
    )


def detect_hand_boxes(
    frame_rgb: np.ndarray,
    previous_rgb: np.ndarray | None,
    next_rgb: np.ndarray | None,
    face_box: tuple[int, int, int, int] | None,
    body_box: tuple[int, int, int, int] | None,
    face_is_estimated: bool = False,
) -> dict[str, tuple[int, int, int, int]]:
    h, w = frame_rgb.shape[:2]
    skin = skin_mask(frame_rgb)
    motion = motion_mask(frame_rgb, previous_rgb, next_rgb)

    mask = skin.copy()

    if body_box is not None:
        roi = np.zeros_like(mask)
        x1, y1, x2, y2 = expand_box(body_box, 0.20, w, h)
        roi[y1:y2, x1:x2] = 255
        mask = cv2.bitwise_and(mask, roi)

    mask[: int(h * 0.05), :] = 0
    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    preferred_mask = mask.copy()
    if motion is not None and np.count_nonzero(motion) > h * w * 0.002:
        preferred_mask = cv2.bitwise_and(mask, dilate_mask(motion, 11))
        preferred_mask = cv2.morphologyEx(preferred_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    min_area = max(18.0, h * w * 0.0006)
    max_area = h * w * 0.20
    face_center_x = (face_box[0] + face_box[2]) / 2.0 if face_box is not None else w / 2.0
    face_area = float(box_area(face_box)) if face_box is not None else 0.0

    candidate_contours: list[tuple[Any, float]] = [(contour, 1.0) for contour in contours]
    preferred_contours, _ = cv2.findContours(preferred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate_contours.extend((contour, 2.4) for contour in preferred_contours)
    if face_box is not None:
        split_mask = mask.copy()
        core = expand_box(face_box, 0.08, w, h)
        cx1, cy1, cx2, cy2 = core
        split_mask[cy1:cy2, cx1:cx2] = 0
        split_contours, _ = cv2.findContours(split_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_contours.extend((contour, 2.1) for contour in split_contours)

    for contour, source_weight in candidate_contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < max(6, int(w * 0.035)) or bh < max(6, int(h * 0.035)):
            continue

        box = expand_box((x, y, x + bw, y + bh), 0.12, w, h)

        cx, cy = box_center(box)
        if body_box is not None:
            _bx1, by1, _bx2, by2 = body_box
            if cy > by1 + (by2 - by1) * 0.96:
                continue

        motion_bonus = 0.0
        if motion is not None:
            bx1, by1, bx2, by2 = box
            motion_bonus = float(np.count_nonzero(motion[by1:by2, bx1:bx2]))

        if (
            face_box is not None
            and intersection_ratio(box, face_box) > 0.70
            and box_area(box) < face_area * 1.65
            and motion_bonus < box_area(box) * 0.10
        ):
            continue

        side_bonus = abs(cx - face_center_x) * 0.15
        upper_bonus = max(0.0, h - cy) * 0.18
        score = (
            np.sqrt(area) * 12.0
            + motion_bonus * 0.055
            + side_bonus
            + upper_bonus
        ) * source_weight
        candidates.append((score, box))

    near_face_box = near_face_hand_box(mask, motion, face_box, w, h, face_is_estimated)
    if near_face_box is not None:
        candidates.append((h * w * 0.95, near_face_box))

    detected = select_distinct_hand_candidates(candidates, w, h)
    selected = complete_hand_boxes(detected, face_box, body_box, w, h)
    selected = [
        refine_hand_box(box, skin, face_box, body_box, w, h)
        for box in selected
    ]
    selected = sorted(selected[:2], key=lambda box: box_center(box)[0])

    if len(selected) < 2:
        return {}

    return {
        "left hand": selected[0],
        "right hand": selected[1],
    }


def refine_hand_box(
    box: tuple[int, int, int, int],
    skin: np.ndarray,
    face_box: tuple[int, int, int, int] | None,
    body_box: tuple[int, int, int, int] | None,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    if box_w < width * 0.34 and box_h < height * 0.34:
        return box

    local = skin[y1:y2, x1:x2]
    ys, xs = np.where(local > 0)
    if len(xs) < 20:
        return box

    xs = xs + x1
    ys = ys + y1

    if face_box is not None:
        fx1, fy1, fx2, fy2 = expand_box(face_box, 0.04, width, height)
        outside_face = ~((fx1 <= xs) & (xs <= fx2) & (fy1 <= ys) & (ys <= fy2))
        if np.count_nonzero(outside_face) >= 20:
            xs = xs[outside_face]
            ys = ys[outside_face]

    near_face = False
    if face_box is not None:
        fx1, fy1, fx2, fy2 = face_box
        face_center_x = (fx1 + fx2) / 2.0
        face_anchor_y = fy1 + (fy2 - fy1) * 0.72
        near_face = y1 < fy2 + (fy2 - fy1) * 2.0
        if near_face:
            distances = (xs - face_center_x) ** 2 + (ys - face_anchor_y) ** 2
            keep = max(20, int(len(xs) * 0.42))
            chosen = np.argpartition(distances, min(keep, len(xs) - 1))[:keep]
            xs = xs[chosen]
            ys = ys[chosen]

    if len(xs) < 20:
        return box

    rx1, ry1 = int(xs.min()), int(ys.min())
    rx2, ry2 = int(xs.max()), int(ys.max())

    if face_box is not None:
        face_w = max(1, face_box[2] - face_box[0])
        face_h = max(1, face_box[3] - face_box[1])
        min_w = int(face_w * (1.05 if near_face else 0.85))
        min_h = int(face_h * (1.00 if near_face else 0.80))
    else:
        min_w = int(width * 0.12)
        min_h = int(height * 0.12)

    refined = ensure_min_box_size((rx1, ry1, rx2, ry2), min_w, min_h, width, height)
    return expand_box(refined, 0.18, width, height)


def select_distinct_hand_candidates(
    candidates: list[tuple[float, tuple[int, int, int, int]]],
    width: int,
    height: int,
) -> list[tuple[int, int, int, int]]:
    selected: list[tuple[int, int, int, int]] = []
    for _score, box in sorted(candidates, key=lambda item: item[0], reverse=True):
        if not is_plausible_hand_box(box, width, height):
            continue
        if any(intersection_ratio(box, kept) > 0.35 for kept in selected):
            continue
        if any(center_distance(box, kept) < min(width, height) * 0.12 for kept in selected):
            continue
        selected.append(box)
        if len(selected) >= 3:
            break
    return selected


def is_plausible_hand_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
) -> bool:
    x1, y1, x2, y2 = box
    box_w = max(0, x2 - x1)
    box_h = max(0, y2 - y1)
    area = box_w * box_h
    if area < width * height * 0.0008:
        return False
    if area > width * height * 0.24:
        return False
    if box_w > width * 0.68 or box_h > height * 0.64:
        return False
    return True


def near_face_hand_box(
    skin: np.ndarray,
    motion: np.ndarray | None,
    face_box: tuple[int, int, int, int] | None,
    width: int,
    height: int,
    face_is_estimated: bool,
) -> tuple[int, int, int, int] | None:
    if face_box is None:
        return None

    fx1, fy1, fx2, fy2 = face_box
    fw = max(1, fx2 - fx1)
    fh = max(1, fy2 - fy1)
    zone = clamp_box(
        (
            fx1 - int(fw * 1.15),
            fy1 + int(fh * 0.12),
            fx2 + int(fw * 1.15),
            fy2 + int(fh * 0.95),
        ),
        width,
        height,
    )
    zx1, zy1, zx2, zy2 = zone
    if zx2 <= zx1 or zy2 <= zy1:
        return None

    zone_mask = np.zeros_like(skin)
    zone_mask[zy1:zy2, zx1:zx2] = 255
    hand_mask = cv2.bitwise_and(skin, zone_mask)

    if motion is not None:
        moving_skin = cv2.bitwise_and(hand_mask, dilate_mask(motion, 9))
        if np.count_nonzero(moving_skin) >= max(18, int(width * height * 0.00045)):
            hand_mask = moving_skin
        elif not face_is_estimated:
            return None
    elif not face_is_estimated:
        return None

    contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    boxes: list[tuple[int, int, int, int]] = []
    min_area = max(12.0, width * height * 0.00035)
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        boxes.append((x, y, x + bw, y + bh))

    if not boxes:
        return None

    merged = boxes[0]
    for box in boxes[1:]:
        merged = union_box(merged, box, width, height)

    return expand_box(merged, 0.18, width, height)


def complete_hand_boxes(
    detected_boxes: list[tuple[int, int, int, int]],
    face_box: tuple[int, int, int, int] | None,
    body_box: tuple[int, int, int, int] | None,
    width: int,
    height: int,
) -> list[tuple[int, int, int, int]]:
    selected: list[tuple[int, int, int, int]] = []
    for box in detected_boxes:
        if is_plausible_hand_box(box, width, height) and all(
            intersection_ratio(box, kept) < 0.45 for kept in selected
        ):
            selected.append(box)
        if len(selected) >= 2:
            return selected

    estimates = estimated_hand_boxes(face_box, body_box, width, height)
    for box in estimates:
        if all(intersection_ratio(box, kept) < 0.40 for kept in selected):
            selected.append(box)
        if len(selected) >= 2:
            return selected

    for box in estimates:
        if all(center_distance(box, kept) > min(width, height) * 0.05 for kept in selected):
            selected.append(box)
        if len(selected) >= 2:
            return selected

    return selected


def estimated_hand_boxes(
    face_box: tuple[int, int, int, int] | None,
    body_box: tuple[int, int, int, int] | None,
    width: int,
    height: int,
) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []

    if body_box is not None:
        bx1, by1, bx2, by2 = body_box
        bw = max(1, bx2 - bx1)
        bh = max(1, by2 - by1)
        boxes.extend(
            [
                clamp_box(
                    (
                        bx1 + int(bw * 0.05),
                        by1 + int(bh * 0.24),
                        bx1 + int(bw * 0.42),
                        by1 + int(bh * 0.58),
                    ),
                    width,
                    height,
                ),
                clamp_box(
                    (
                        bx1 + int(bw * 0.58),
                        by1 + int(bh * 0.24),
                        bx1 + int(bw * 0.95),
                        by1 + int(bh * 0.58),
                    ),
                    width,
                    height,
                ),
                clamp_box(
                    (
                        bx1 + int(bw * 0.28),
                        by1 + int(bh * 0.55),
                        bx1 + int(bw * 0.78),
                        by1 + int(bh * 0.84),
                    ),
                    width,
                    height,
                ),
            ]
        )

    if face_box is not None:
        fx1, fy1, fx2, fy2 = face_box
        fw = max(1, fx2 - fx1)
        fh = max(1, fy2 - fy1)
        boxes.append(
            clamp_box(
                (
                    fx1 - int(fw * 0.70),
                    fy1 + int(fh * 0.15),
                    fx2 + int(fw * 0.70),
                    fy2 + int(fh * 0.60),
                ),
                width,
                height,
            )
        )

    if not boxes:
        boxes = [
            (int(width * 0.12), int(height * 0.28), int(width * 0.42), int(height * 0.62)),
            (int(width * 0.58), int(height * 0.28), int(width * 0.88), int(height * 0.62)),
        ]

    return boxes


def skin_mask(frame_rgb: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)
    mask_ycrcb = cv2.inRange(
        ycrcb,
        np.array([0, 133, 77], dtype=np.uint8),
        np.array([255, 173, 127], dtype=np.uint8),
    )

    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    mask_hsv_1 = cv2.inRange(
        hsv,
        np.array([0, 35, 45], dtype=np.uint8),
        np.array([25, 255, 255], dtype=np.uint8),
    )
    mask_hsv_2 = cv2.inRange(
        hsv,
        np.array([160, 35, 45], dtype=np.uint8),
        np.array([180, 255, 255], dtype=np.uint8),
    )
    return cv2.bitwise_or(mask_ycrcb, cv2.bitwise_or(mask_hsv_1, mask_hsv_2))


def motion_mask(
    frame_rgb: np.ndarray,
    previous_rgb: np.ndarray | None,
    next_rgb: np.ndarray | None = None,
) -> np.ndarray | None:
    references = [
        reference
        for reference in (previous_rgb, next_rgb)
        if reference is not None and reference.shape == frame_rgb.shape
    ]
    if not references:
        return None

    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    for reference in references:
        ref_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(gray, ref_gray)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        _, current_mask = cv2.threshold(diff, 8, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_or(mask, current_mask)
    return dilate_mask(mask, 9)


def dilate_mask(mask: np.ndarray, size: int) -> np.ndarray:
    kernel = np.ones((size, size), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def foreground_bbox(frame_rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    h, w = frame_rgb.shape[:2]
    frame_float = frame_rgb.astype(np.float32)
    strip = max(3, int(min(h, w) * 0.04))
    border_pixels = np.concatenate(
        [
            frame_float[:strip, :, :].reshape(-1, 3),
            frame_float[-strip:, :, :].reshape(-1, 3),
            frame_float[:, :strip, :].reshape(-1, 3),
            frame_float[:, -strip:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    background_color = np.median(border_pixels, axis=0)
    distance = np.linalg.norm(frame_float - background_color.reshape(1, 1, 3), axis=2)
    mask = (distance > 28.0).astype(np.uint8) * 255

    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    vivid_green = cv2.inRange(hsv, np.array([35, 45, 40]), np.array([95, 255, 255]))
    vivid_yellow = cv2.inRange(hsv, np.array([15, 45, 80]), np.array([45, 255, 255]))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(cv2.bitwise_or(vivid_green, vivid_yellow)))

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    boxes: list[tuple[int, int, int, int]] = []
    min_area = max(80.0, h * w * 0.006)
    for contour in contours[:5]:
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        boxes.append((x, y, x + bw, y + bh))

    if not boxes:
        return None

    person = boxes[0]
    for box in boxes[1:]:
        if center_distance(person, box) < min(h, w) * 0.45:
            person = union_box(person, box, w, h)

    person = expand_box(person, 0.06, w, h)
    if box_area(person) > h * w * 0.78:
        return None
    return person


def expand_box(
    box: tuple[int, int, int, int],
    ratio: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    pad_x = int((x2 - x1) * ratio)
    pad_y = int((y2 - y1) * ratio)
    return clamp_box((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), width, height)


def clamp_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        int(np.clip(x1, 0, width - 1)),
        int(np.clip(y1, 0, height - 1)),
        int(np.clip(x2, 0, width - 1)),
        int(np.clip(y2, 0, height - 1)),
    )


def ensure_min_box_size(
    box: tuple[int, int, int, int],
    min_width: int,
    min_height: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    half_w = max(max(1, x2 - x1) // 2, int(min_width) // 2)
    half_h = max(max(1, y2 - y1) // 2, int(min_height) // 2)
    return clamp_box((cx - half_w, cy - half_h, cx + half_w, cy + half_h), width, height)


def box_area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def center_distance(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    ax, ay = box_center(first)
    bx, by = box_center(second)
    return float(np.hypot(ax - bx, ay - by))


def union_box(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    return clamp_box(
        (
            min(first[0], second[0]),
            min(first[1], second[1]),
            max(first[2], second[2]),
            max(first[3], second[3]),
        ),
        width,
        height,
    )


def intersection_ratio(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    inter = box_area((x1, y1, x2, y2))
    smaller = max(1, min(box_area(first), box_area(second)))
    return inter / smaller


def draw_box(
    frame_bgr: np.ndarray,
    box: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
) -> None:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box
    x1 = int(np.clip(x1, 0, w - 1))
    x2 = int(np.clip(x2, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    y2 = int(np.clip(y2, 0, h - 1))
    if x2 <= x1 or y2 <= y1:
        return

    thickness = max(1, int(round(min(h, w) / 140)))
    font_scale = max(0.38, min(0.58, min(h, w) / 430))
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness)
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    label_w, label_h = label_size
    label_y1 = max(0, y1 - label_h - 8)
    cv2.rectangle(
        frame_bgr,
        (x1, label_y1),
        (min(w - 1, x1 + label_w + 10), y1),
        color,
        -1,
    )
    cv2.putText(
        frame_bgr,
        label,
        (x1 + 5, max(label_h + 1, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (20, 20, 20),
        thickness,
        cv2.LINE_AA,
    )


def write_browser_mp4(
    frames_bgr: list[np.ndarray],
    fps: float,
    output_path: Path,
) -> None:
    if not frames_bgr:
        raise ValueError("Không có frame để ghi video.")

    height, width = frames_bgr[0].shape[:2]
    fps = max(1.0, min(float(fps), 60.0))

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.3f}",
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    import subprocess

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    stderr = b""
    try:
        assert process.stdin is not None
        for frame_bgr in frames_bgr:
            if frame_bgr.shape[:2] != (height, width):
                frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
            process.stdin.write(np.ascontiguousarray(frame_bgr).tobytes())
        process.stdin.close()
        _stdout, stderr = process.communicate(timeout=120)
    except subprocess.TimeoutExpired as error:
        process.kill()
        raise RuntimeError("Encode video mất quá lâu.") from error
    finally:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()

    if process.returncode != 0:
        message = stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"Không encode được MP4 H.264: {message[-600:]}")


def extract_video_path(value: Any) -> Path:
    if value is None:
        raise ValueError("Hãy upload một video .mp4 trước.")

    if isinstance(value, dict):
        value = value.get("path") or value.get("name")

    path = Path(str(value))
    if not path.exists() or not path.is_file():
        raise ValueError("File video không tồn tại hoặc không truy cập được.")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError("Demo này chỉ nhận file .mp4.")
    return path


def run_demo(video_file: Any) -> tuple[str | None, str]:
    try:
        video_path = extract_video_path(video_file)
        visual_path = create_bbox_visualization(video_path)
        label, confidence = PREDICTOR.predict(video_path)
        prediction = (
            f"<div class='prediction-card'>"
            f"<div class='prediction-title'>Predict label</div>"
            f"<div class='prediction-label'>{label}</div>"
            f"<div class='prediction-confidence'>{confidence:.2f}%</div>"
            f"</div>"
        )
        return visual_path, prediction
    except Exception as error:
        LOGGER.exception("Demo failed")
        message = str(error)
        prediction = (
            "<div class='prediction-card prediction-error'>"
            "<div class='prediction-title'>Predict label</div>"
            "<div class='prediction-label'>Chưa dự đoán được</div>"
            f"<div class='prediction-confidence'>{message}</div>"
            "</div>"
        )
        return None, prediction


def clear_demo() -> tuple[None, str]:
    return None, empty_prediction()


def empty_prediction() -> str:
    return (
        "<div class='prediction-card prediction-empty'>"
        "<div class='prediction-title'>Predict label</div>"
        "<div class='prediction-label'>Waiting</div>"
        "<div class='prediction-confidence'>0.00%</div>"
        "</div>"
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Sign Language") as demo:
        gr.HTML(
            """
            <h1 class="app-title">Sign Language</h1>
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                video_input = gr.File(
                    label="Input video (.mp4)",
                    file_types=[".mp4"],
                    type="filepath",
                )
                with gr.Row():
                    predict_button = gr.Button("Predict", variant="primary", size="lg")
                    clear_button = gr.Button("Clear", size="lg")

            with gr.Column(scale=1):
                prediction_output = gr.HTML(empty_prediction())

        visual_output = gr.Video(
            label="Video visual + bounding boxes",
            interactive=False,
            height=520,
        )

        predict_button.click(
            fn=run_demo,
            inputs=[video_input],
            outputs=[visual_output, prediction_output],
            show_progress="full",
        )
        clear_button.click(
            fn=clear_demo,
            inputs=[],
            outputs=[visual_output, prediction_output],
            queue=False,
        )

    return demo


def demo_css() -> str:
    return """
    .gradio-container {
      max-width: 1080px !important;
      margin: 0 auto !important;
    }
    .app-title {
      margin: 10px 0 24px;
      color: var(--body-text-color);
      font-size: clamp(38px, 7vw, 66px);
      line-height: 1.05;
      letter-spacing: -.045em;
      font-weight: 850;
    }
    .prediction-card {
      min-height: 210px;
      padding: 28px;
      border-radius: 24px;
      border: 1px solid rgba(234, 88, 12, .22);
      background: linear-gradient(145deg, rgba(234, 88, 12, .13), rgba(249, 115, 22, .07));
      display: flex;
      flex-direction: column;
      justify-content: center;
    }
    .prediction-empty {
      filter: saturate(.55);
    }
    .prediction-error {
      border-color: rgba(220, 38, 38, .32);
      background: linear-gradient(145deg, rgba(220, 38, 38, .12), rgba(249, 115, 22, .08));
    }
    .prediction-title {
      color: #c2410c;
      font-size: 13px;
      font-weight: 800;
      letter-spacing: .12em;
      text-transform: uppercase;
    }
    .prediction-label {
      margin-top: 12px;
      font-size: clamp(34px, 5vw, 58px);
      font-weight: 850;
      line-height: 1.05;
    }
    .prediction-confidence {
      margin-top: 14px;
      color: #ea580c;
      font-size: 24px;
      font-weight: 800;
      overflow-wrap: anywhere;
    }
    """


if __name__ == "__main__":
    demo = build_app().queue(default_concurrency_limit=1)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=False,
        prevent_thread_lock=True,
        show_error=True,
        css=demo_css(),
    )
    threading.Event().wait()
