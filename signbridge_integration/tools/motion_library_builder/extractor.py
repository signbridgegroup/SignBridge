"""Reusable wrappers around the approved extraction format.

``legacy_extractor.py`` is a byte-for-byte copy of
``C:\\SignBridge_Project\\src\\extract_sign_motion.py`` — confirmed to be the
script that actually produced the 7 approved ``public/motions/*.json``
files (exact top-level key match, no ``"processing"`` block). Its helper
functions are reused directly here so every newly generated motion file is
written in the identical approved ``signbridge-motion-v1`` schema. This
module adds only the plumbing to (a) drive it from parameters instead of
argparse and (b) read an ordered image-frame sequence instead of a video
file, for Isharah.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

import legacy_extractor as _legacy

_NUMBER_RE = re.compile(r"(\d+)")


def natural_sort_key(path: Path) -> tuple:
    parts = _NUMBER_RE.split(path.name)
    return tuple(int(p) if p.isdigit() else p for p in parts)


def _build_landmarker(model_path: Path, min_hand_confidence: float):
    options = mp.tasks.vision.HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=min_hand_confidence,
    )
    return mp.tasks.vision.HolisticLandmarker.create_from_options(options)


def _assemble_and_write(
    *,
    output: Path,
    name: str,
    fps: float,
    source_file: str,
    start_frame: int,
    rotation: str,
    width: int,
    height: int,
    finger_mode: str,
    smoothing_passes: int,
    pose,
    pose_world,
    left_hand,
    left_hand_world,
    right_hand,
    right_hand_world,
    left_detected,
    right_detected,
    pose_detected,
    face_weights,
    face_names,
) -> dict[str, Any]:
    if not pose:
        raise RuntimeError("No frames were processed")

    pose_array = _legacy.fill_and_smooth(np.stack(pose), smoothing_passes)
    pose_world_array = _legacy.fill_and_smooth(np.stack(pose_world), smoothing_passes)
    left_array = _legacy.fill_and_smooth(np.stack(left_hand), smoothing_passes)
    left_world_array = _legacy.fill_and_smooth(np.stack(left_hand_world), smoothing_passes)
    right_array = _legacy.fill_and_smooth(np.stack(right_hand), smoothing_passes)
    right_world_array = _legacy.fill_and_smooth(np.stack(right_hand_world), smoothing_passes)

    weights_array = np.asarray(face_weights, dtype=np.float32)
    if weights_array.ndim == 2 and weights_array.shape[1]:
        weights_array = _legacy.fill_and_smooth_weights(weights_array, smoothing_passes)

    frames = []
    for index in range(len(pose_array)):
        frames.append(
            {
                "pose": _legacy.rounded(pose_array[index]),
                "poseWorld": _legacy.rounded(pose_world_array[index]),
                "leftHand": _legacy.rounded(left_array[index]),
                "leftHandWorld": _legacy.rounded(left_world_array[index]),
                "rightHand": _legacy.rounded(right_array[index]),
                "rightHandWorld": _legacy.rounded(right_world_array[index]),
                "face": (
                    _legacy.rounded(weights_array[index], 5)
                    if weights_array.ndim == 2 and weights_array.shape[1]
                    else []
                ),
                "poseDetected": bool(pose_detected[index]),
                "leftHandDetected": bool(left_detected[index]),
                "rightHandDetected": bool(right_detected[index]),
            }
        )

    payload = {
        "format": "signbridge-motion-v1",
        "name": name,
        "fps": round(fps, 6),
        "frameCount": len(frames),
        "durationSeconds": round(len(frames) / fps, 6),
        "fingerMode": finger_mode,
        "source": {
            "file": source_file,
            "startFrame": start_frame,
            "endFrame": start_frame + len(frames) - 1,
            "rotation": rotation,
            "width": width,
            "height": height,
        },
        "detection": {
            "poseRate": round(float(np.mean(pose_detected)), 6),
            "leftHandRate": round(float(np.mean(left_detected)), 6),
            "rightHandRate": round(float(np.mean(right_detected)), 6),
        },
        "faceBlendshapeNames": face_names or [],
        "frames": frames,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(output)

    return {
        "frame_count": len(frames),
        "fps": fps,
        "pose_rate": payload["detection"]["poseRate"],
        "left_hand_rate": payload["detection"]["leftHandRate"],
        "right_hand_rate": payload["detection"]["rightHandRate"],
    }


def extract_from_video(
    *,
    video: Path,
    output: Path,
    name: str,
    model: Path,
    start_frame: int = 0,
    end_frame: int | None = None,
    rotation: str = "none",
    min_hand_confidence: float = 0.25,
    smoothing_passes: int = 2,
    finger_mode: str = "tracked",
) -> dict[str, Any]:
    if not video.is_file():
        raise FileNotFoundError(video)
    if not model.is_file():
        raise FileNotFoundError(model)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    resolved_end = frame_count - 1 if end_frame is None else min(end_frame, frame_count - 1)
    if resolved_end < start_frame:
        raise ValueError("end-frame must be >= start-frame")

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    pose, pose_world = [], []
    left_hand, left_hand_world = [], []
    right_hand, right_hand_world = [], []
    face_weights: list[list[float]] = []
    face_names: list[str] | None = None
    left_detected, right_detected, pose_detected = [], [], []
    width = height = 0

    try:
        with _build_landmarker(model, min_hand_confidence) as landmarker:
            output_index = 0
            for _ in range(start_frame, resolved_end + 1):
                ok, frame = capture.read()
                if not ok:
                    break
                frame = _legacy.rotate_frame(frame, rotation)
                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(round(output_index * 1000.0 / fps))
                result = landmarker.detect_for_video(image, timestamp_ms)
                face_names = _accumulate(
                    result, pose, pose_world, left_hand, left_hand_world,
                    right_hand, right_hand_world, left_detected, right_detected,
                    pose_detected, face_weights, face_names,
                )
                output_index += 1
    finally:
        capture.release()

    return _assemble_and_write(
        output=output, name=name, fps=fps, source_file=video.name,
        start_frame=start_frame, rotation=rotation, width=width, height=height,
        finger_mode=finger_mode, smoothing_passes=smoothing_passes,
        pose=pose, pose_world=pose_world, left_hand=left_hand,
        left_hand_world=left_hand_world, right_hand=right_hand,
        right_hand_world=right_hand_world, left_detected=left_detected,
        right_detected=right_detected, pose_detected=pose_detected,
        face_weights=face_weights, face_names=face_names,
    )


def extract_from_image_sequence(
    *,
    frames_dir: Path,
    output: Path,
    name: str,
    model: Path,
    relative_start_frame: int,
    relative_end_frame: int,
    fps: float = 30.0,
    min_hand_confidence: float = 0.25,
    smoothing_passes: int = 2,
    finger_mode: str = "tracked",
) -> dict[str, Any]:
    """Feed an ordered Isharah image sequence through the same MediaPipe path.

    ``relative_end_frame`` is EXCLUSIVE, matching the [start, end) convention
    used throughout ``extract_isharah_gloss_library.py`` (its own clip
    extraction is a plain Python slice: ``source_features[start:end]``).
    """
    if not frames_dir.is_dir():
        raise FileNotFoundError(frames_dir)
    if not model.is_file():
        raise FileNotFoundError(model)

    all_images = sorted(frames_dir.glob("*.jpg"), key=natural_sort_key)
    if not all_images:
        all_images = sorted(frames_dir.glob("*.png"), key=natural_sort_key)
    if not all_images:
        raise FileNotFoundError(f"No frame images found in {frames_dir}")

    if relative_end_frame > len(all_images):
        raise ValueError(
            f"relative_end_frame {relative_end_frame} exceeds available frames "
            f"({len(all_images)}) in {frames_dir}"
        )

    selected = all_images[relative_start_frame:relative_end_frame]
    if not selected:
        raise ValueError("Selected Isharah frame range is empty")

    pose, pose_world = [], []
    left_hand, left_hand_world = [], []
    right_hand, right_hand_world = [], []
    face_weights: list[list[float]] = []
    face_names: list[str] | None = None
    left_detected, right_detected, pose_detected = [], [], []
    width = height = 0

    try:
        with _build_landmarker(model, min_hand_confidence) as landmarker:
            for output_index, image_path in enumerate(selected):
                frame = cv2.imread(str(image_path))
                if frame is None:
                    raise RuntimeError(f"Could not read image: {image_path}")
                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(round(output_index * 1000.0 / fps))
                result = landmarker.detect_for_video(image, timestamp_ms)
                face_names = _accumulate(
                    result, pose, pose_world, left_hand, left_hand_world,
                    right_hand, right_hand_world, left_detected, right_detected,
                    pose_detected, face_weights, face_names,
                )
    finally:
        pass  # no VideoCapture to release

    return _assemble_and_write(
        output=output, name=name, fps=fps, source_file=frames_dir.name,
        start_frame=relative_start_frame, rotation="none", width=width, height=height,
        finger_mode=finger_mode, smoothing_passes=smoothing_passes,
        pose=pose, pose_world=pose_world, left_hand=left_hand,
        left_hand_world=left_hand_world, right_hand=right_hand,
        right_hand_world=right_hand_world, left_detected=left_detected,
        right_detected=right_detected, pose_detected=pose_detected,
        face_weights=face_weights, face_names=face_names,
    )


def _accumulate(
    result, pose, pose_world, left_hand, left_hand_world, right_hand,
    right_hand_world, left_detected, right_detected, pose_detected, face_weights,
    face_names: list[str] | None,
) -> list[str] | None:
    """Mirrors extract_sign_motion.py's per-frame accumulation exactly,
    including its ordering: face_names is fixed from the first frame that
    sets it (even if that frame had no detected face), then every frame's
    row — including that same first frame — is built against that fixed
    name list, matching the approved script byte-for-byte in behaviour.
    """
    pose_ok = len(result.pose_landmarks) == 33 and len(result.pose_world_landmarks) == 33
    left_ok = len(result.left_hand_landmarks) == 21 and len(result.left_hand_world_landmarks) == 21
    right_ok = len(result.right_hand_landmarks) == 21 and len(result.right_hand_world_landmarks) == 21

    pose_detected.append(pose_ok)
    left_detected.append(left_ok)
    right_detected.append(right_ok)

    pose.append(_legacy.landmark_values(result.pose_landmarks, 33))
    pose_world.append(_legacy.landmark_values(result.pose_world_landmarks, 33))
    left_hand.append(_legacy.landmark_values(result.left_hand_landmarks, 21))
    left_hand_world.append(_legacy.landmark_values(result.left_hand_world_landmarks, 21))
    right_hand.append(_legacy.landmark_values(result.right_hand_landmarks, 21))
    right_hand_world.append(_legacy.landmark_values(result.right_hand_world_landmarks, 21))

    categories = result.face_blendshapes or []
    current = {c.category_name: c.score for c in categories}
    if face_names is None:
        face_names = [c.category_name for c in categories if c.category_name != "_neutral"]
    face_weights.append([current.get(name, 0.0) for name in (face_names or [])])
    return face_names
