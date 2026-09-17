from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

from finetune_jordanian_it import TransferClassifier, to_shared198
from scripts.prepare_model_sequences import prepare_sequence, record_key


PROJECT = Path(r"C:\SignBridge_Project")
HOLISTIC_MODEL = PROJECT / "models" / "holistic_landmarker.task"
ACTIVE_REPORT = PROJECT / "data" / "processed" / "active_length_report.csv"
CHECKPOINT = (
    PROJECT / "data" / "processed" / "model_data"
    / "training_results_transfer" / "best_transfer_model.pt"
)

FACE_POINTS, POSE_POINTS, HAND_POINTS = 468, 33, 21


def arguments():
    parser = argparse.ArgumentParser(description="Predict an IT sign directly from a video")
    parser.add_argument("--video", type=Path, required=True)
    return parser.parse_args()


def landmark_array(landmarks, expected):
    output = np.zeros((expected, 3), dtype=np.float32)
    for index, landmark in enumerate(landmarks[:expected]):
        output[index] = (landmark.x, landmark.y, landmark.z)
    return output.reshape(-1)


def extract_video(video_path: Path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    options = mp.tasks.vision.HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(HOLISTIC_MODEL)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    frames = []
    frame_number = 0
    try:
        with mp.tasks.vision.HolisticLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(image, int(frame_number * 1000 / fps))
                frames.append(np.concatenate([
                    landmark_array(result.face_landmarks, FACE_POINTS),
                    landmark_array(result.pose_landmarks, POSE_POINTS),
                    landmark_array(result.left_hand_landmarks, HAND_POINTS),
                    landmark_array(result.right_hand_landmarks, HAND_POINTS),
                ]))
                frame_number += 1
    finally:
        capture.release()
    if not frames:
        raise RuntimeError("No frames were extracted from the video")
    return np.stack(frames).astype(np.float32)


def active_bounds(video_path: Path, frame_count: int):
    if not ACTIVE_REPORT.is_file():
        return 0, frame_count - 1, False
    wanted = record_key(video_path.parent.name, video_path.name)
    with ACTIVE_REPORT.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if record_key(row["label"], row["video"]) == wanted:
                return int(row["start_frame"]), int(row["end_frame"]), True
    return 0, frame_count - 1, False


def main():
    args = arguments()
    video = args.video.resolve()
    for required in (video, HOLISTIC_MODEL, CHECKPOINT):
        if not required.is_file():
            raise FileNotFoundError(required)

    print("Extracting MediaPipe landmarks from:", video)
    raw = extract_video(video)
    start, end, used_report = active_bounds(video, len(raw))
    sequence, stats = prepare_sequence(raw, start, end, 200, 5)
    shared = to_shared198(sequence)
    velocity = np.zeros_like(shared)
    velocity[1:] = shared[1:] - shared[:-1]
    features = np.concatenate([shared, velocity], axis=1)

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    labels = checkpoint["labels"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransferClassifier(len(labels)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    tensor = torch.from_numpy(features).unsqueeze(0).to(device)
    lengths = torch.tensor([len(features)], dtype=torch.long)
    with torch.no_grad():
        probabilities = model(tensor, lengths).softmax(1)[0]
    values, indices = probabilities.topk(5)

    print("Frames:", len(raw))
    print("Active segment:", f"{start}:{end}", "(saved report)" if used_report else "(full video)")
    print("Left/right hand observed:",
          f"{stats['resampled_left_observed_rate']:.2f}% / {stats['resampled_right_observed_rate']:.2f}%")
    print("Top-5 IT predictions:")
    for rank, (index, probability) in enumerate(zip(indices.tolist(), values.tolist()), 1):
        print(f"{rank}. {labels[index]}: {probability * 100:.2f}%")


if __name__ == "__main__":
    main()
