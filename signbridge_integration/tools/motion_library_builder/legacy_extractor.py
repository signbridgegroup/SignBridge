"""Extract animation-ready MediaPipe motion from one sign-language video.

The output is deliberately model-independent: it stores smoothed 3D pose,
hand landmarks and ARKit-compatible face blendshape weights in one JSON file.
The SignBridge Three.js viewer retargets that JSON to the MetaPerson rig.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


DEFAULT_PROJECT = Path(r"C:\SignBridge_Project")
DEFAULT_MODEL = DEFAULT_PROJECT / "models" / "holistic_landmarker.task"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a SignBridge avatar motion JSON from a video."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default="Sign")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument(
        "--rotation",
        choices=("none", "clockwise", "counterclockwise", "180"),
        default="none",
        help="Apply this rotation before MediaPipe processing.",
    )
    parser.add_argument("--min-hand-confidence", type=float, default=0.25)
    parser.add_argument("--smoothing-passes", type=int, default=2)
    parser.add_argument(
        "--finger-mode",
        choices=("tracked", "open"),
        default="tracked",
        help="Use tracked finger articulation or keep the avatar's open-hand rest shape.",
    )
    return parser.parse_args()


def rotate_frame(frame: np.ndarray, rotation: str) -> np.ndarray:
    if rotation == "clockwise":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == "counterclockwise":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def landmark_values(landmarks: list, count: int) -> np.ndarray:
    values = np.full((count, 3), np.nan, dtype=np.float32)
    for index, landmark in enumerate(landmarks[:count]):
        values[index] = (landmark.x, landmark.y, landmark.z)
    return values


def fill_and_smooth(values: np.ndarray, passes: int) -> np.ndarray:
    """Interpolate missing detections and apply a small zero-phase low-pass."""
    output = np.asarray(values, dtype=np.float32).copy()
    frame_axis = np.arange(len(output), dtype=np.float32)

    for point in range(output.shape[1]):
        for coordinate in range(output.shape[2]):
            series = output[:, point, coordinate]
            valid = np.isfinite(series)
            if not np.any(valid):
                series[:] = 0.0
            elif np.count_nonzero(valid) == 1:
                series[:] = series[valid][0]
            else:
                series[:] = np.interp(frame_axis, frame_axis[valid], series[valid])

    kernel = np.asarray([1, 2, 3, 2, 1], dtype=np.float32) / 9.0
    for _ in range(max(0, passes)):
        padded = np.pad(output, ((2, 2), (0, 0), (0, 0)), mode="edge")
        output = sum(
            kernel[index] * padded[index : index + len(output)]
            for index in range(len(kernel))
        )
    return output.astype(np.float32)


def fill_and_smooth_weights(values: np.ndarray, passes: int) -> np.ndarray:
    shaped = values[:, :, None]
    return fill_and_smooth(shaped, passes)[:, :, 0]


def rounded(values: np.ndarray, digits: int = 6) -> list:
    return np.round(values, digits).tolist()


def main() -> None:
    args = arguments()
    video = args.video.resolve()
    model = args.model.resolve()
    output = args.output.resolve()

    if not video.is_file():
        raise FileNotFoundError(video)
    if not model.is_file():
        raise FileNotFoundError(model)
    if args.start_frame < 0:
        raise ValueError("start-frame cannot be negative")
    if not 0.0 <= args.min_hand_confidence <= 1.0:
        raise ValueError("min-hand-confidence must be between 0 and 1")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = frame_count - 1 if args.end_frame is None else args.end_frame
    end_frame = min(end_frame, frame_count - 1)
    if end_frame < args.start_frame:
        raise ValueError("end-frame must be greater than or equal to start-frame")

    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    options = mp.tasks.vision.HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=args.min_hand_confidence,
        # Keep this disabled during offline clip extraction. Some MediaPipe
        # builds abort when a transient empty face packet is requested while
        # the hands enter the frame. Natural blinking remains viewer-driven.
        output_face_blendshapes=False,
    )

    pose = []
    pose_world = []
    left_hand = []
    left_hand_world = []
    right_hand = []
    right_hand_world = []
    face_weights = []
    face_names: list[str] | None = None
    left_detected = []
    right_detected = []
    pose_detected = []
    width = height = 0

    try:
        with mp.tasks.vision.HolisticLandmarker.create_from_options(options) as landmarker:
            output_index = 0
            for frame_number in range(args.start_frame, end_frame + 1):
                ok, frame = capture.read()
                if not ok:
                    break
                frame = rotate_frame(frame, args.rotation)
                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(round(output_index * 1000.0 / fps))
                result = landmarker.detect_for_video(image, timestamp_ms)

                pose_ok = len(result.pose_landmarks) == 33
                left_ok = len(result.left_hand_landmarks) == 21
                right_ok = len(result.right_hand_landmarks) == 21
                pose_detected.append(pose_ok)
                left_detected.append(left_ok)
                right_detected.append(right_ok)

                pose.append(landmark_values(result.pose_landmarks, 33))
                pose_world.append(landmark_values(result.pose_world_landmarks, 33))
                left_hand.append(landmark_values(result.left_hand_landmarks, 21))
                left_hand_world.append(
                    landmark_values(result.left_hand_world_landmarks, 21)
                )
                right_hand.append(landmark_values(result.right_hand_landmarks, 21))
                right_hand_world.append(
                    landmark_values(result.right_hand_world_landmarks, 21)
                )

                categories = result.face_blendshapes or []
                current = {category.category_name: category.score for category in categories}
                if face_names is None:
                    face_names = [
                        category.category_name
                        for category in categories
                        if category.category_name != "_neutral"
                    ]
                face_weights.append([current.get(name, 0.0) for name in face_names or []])
                output_index += 1
    finally:
        capture.release()

    if not pose:
        raise RuntimeError("No video frames were processed")

    pose_array = fill_and_smooth(np.stack(pose), args.smoothing_passes)
    pose_world_array = fill_and_smooth(
        np.stack(pose_world), args.smoothing_passes
    )
    left_array = fill_and_smooth(np.stack(left_hand), args.smoothing_passes)
    left_world_array = fill_and_smooth(
        np.stack(left_hand_world), args.smoothing_passes
    )
    right_array = fill_and_smooth(np.stack(right_hand), args.smoothing_passes)
    right_world_array = fill_and_smooth(
        np.stack(right_hand_world), args.smoothing_passes
    )
    weights_array = np.asarray(face_weights, dtype=np.float32)
    if weights_array.shape[1]:
        weights_array = fill_and_smooth_weights(
            weights_array, args.smoothing_passes
        )

    frames = []
    for index in range(len(pose_array)):
        frames.append(
            {
                "pose": rounded(pose_array[index]),
                "poseWorld": rounded(pose_world_array[index]),
                "leftHand": rounded(left_array[index]),
                "leftHandWorld": rounded(left_world_array[index]),
                "rightHand": rounded(right_array[index]),
                "rightHandWorld": rounded(right_world_array[index]),
                "face": rounded(weights_array[index], 5),
                "poseDetected": bool(pose_detected[index]),
                "leftHandDetected": bool(left_detected[index]),
                "rightHandDetected": bool(right_detected[index]),
            }
        )

    payload = {
        "format": "signbridge-motion-v1",
        "name": args.name,
        "fps": round(fps, 6),
        "frameCount": len(frames),
        "durationSeconds": round(len(frames) / fps, 6),
        "fingerMode": args.finger_mode,
        "source": {
            "file": video.name,
            "startFrame": args.start_frame,
            "endFrame": args.start_frame + len(frames) - 1,
            "rotation": args.rotation,
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
    output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    print("SignBridge motion extraction completed.")
    print("Name:", args.name)
    print("Frames:", len(frames))
    print("FPS:", f"{fps:.3f}")
    print("Pose detected:", f"{np.mean(pose_detected) * 100:.2f}%")
    print("Left hand detected:", f"{np.mean(left_detected) * 100:.2f}%")
    print("Right hand detected:", f"{np.mean(right_detected) * 100:.2f}%")
    print("Output:", output)


if __name__ == "__main__":
    main()
