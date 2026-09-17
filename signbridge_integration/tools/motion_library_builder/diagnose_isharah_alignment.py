"""Diagnose alignment between one Isharah pose pose sequence and its JPG frames.

This tool is deliberately read-only with respect to datasets, manifests, models,
and motion files.  It writes only a JSON report and a JPG contact sheet under
``signbridge_integration/debug``.  It does not retrain the recognizer and does
not change the frozen avatar retargeter.

Default test case:
    sample 00_0355, gloss "ا"

The test compares the stored 2-D Isharah pose timeline with fresh MediaPipe
landmarks extracted from every JPG.  Dynamic Time Warping (DTW) is used only as
a diagnostic proposal for the frame correspondence; the result is never
written into the production manifest automatically.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
INTEGRATION_ROOT = SCRIPT_PATH.parents[2]

DEFAULT_SAMPLE_ID = "00_0355"
DEFAULT_GLOSS = "ا"
DEFAULT_POSE_DIR = PROJECT_ROOT / "data" / "processed" / "isharah1000_pose"
DEFAULT_CLIPS_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "isharah_gloss_clip_library"
    / "clips.csv"
)
DEFAULT_FRAMES_ROOT = (
    PROJECT_ROOT / "data" / "external" / "isharah" / "frames" / "00"
)
DEFAULT_MODEL = PROJECT_ROOT / "models" / "holistic_landmarker.task"
DEFAULT_OUTPUT_ROOT = INTEGRATION_ROOT / "debug"

NUMBER_RE = re.compile(r"(\d+)")
SPATIAL_FEATURES = 196


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose one Isharah pose/JPG temporal alignment safely."
    )
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--gloss", default=DEFAULT_GLOSS)
    parser.add_argument("--pose-dir", type=Path, default=DEFAULT_POSE_DIR)
    parser.add_argument("--clips-csv", type=Path, default=DEFAULT_CLIPS_CSV)
    parser.add_argument("--frames-root", type=Path, default=DEFAULT_FRAMES_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--min-hand-confidence", type=float, default=0.25)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def natural_sort_key(path: Path) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in NUMBER_RE.split(path.name)
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def find_sample_row(samples_csv: Path, sample_id: str) -> dict[str, str]:
    matches = [row for row in read_csv(samples_csv) if row.get("sample_id") == sample_id]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one samples.csv row for {sample_id}; found {len(matches)}"
        )
    return matches[0]


def find_clip_row(clips_csv: Path, sample_id: str, gloss: str) -> dict[str, str]:
    matches = [
        row
        for row in read_csv(clips_csv)
        if row.get("sample_id") == sample_id and row.get("gloss") == gloss
    ]
    if not matches:
        raise ValueError(f"No clips.csv row found for sample={sample_id}, gloss={gloss!r}")
    matches.sort(
        key=lambda row: (
            row.get("review_required") == "True",
            row.get("sample_exact_match") != "True",
            -float(row.get("candidate_score", "0") or 0),
        )
    )
    return matches[0]


def load_source_sequence(pose_dir: Path, sample_id: str) -> tuple[np.ndarray, dict[str, str]]:
    samples_csv = pose_dir / "samples.csv"
    keypoints_path = pose_dir / "keypoints_flat.npy"
    require_file(samples_csv, "Isharah pose samples.csv")
    require_file(keypoints_path, "Isharah keypoints_flat.npy")

    row = find_sample_row(samples_csv, sample_id)
    start = int(row["start"])
    end = int(row["end"])
    if end <= start:
        raise ValueError(f"Invalid source range [{start}, {end}) for {sample_id}")

    flat = np.load(keypoints_path, mmap_mode="r")
    if flat.ndim != 3 or flat.shape[1:] != (86, 2):
        raise ValueError(f"Unexpected keypoints shape: {flat.shape}; expected (*, 86, 2)")
    if end > len(flat):
        raise ValueError(f"Sample end {end} exceeds keypoints length {len(flat)}")
    return np.asarray(flat[start:end], dtype=np.float32), row


def list_images(frames_dir: Path) -> list[Path]:
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Missing image-sequence directory: {frames_dir}")
    images: list[Path] = []
    for suffix in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(frames_dir.glob(suffix))
    images = sorted(set(images), key=natural_sort_key)
    if not images:
        raise FileNotFoundError(f"No JPG/PNG images found in {frames_dir}")
    return images


def extract_jpg_keypoints(
    image_paths: list[Path], model_path: Path, min_hand_confidence: float
) -> tuple[np.ndarray, list[dict[str, Any]], tuple[int, int]]:
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "This script needs the project's existing opencv and mediapipe packages. "
            "Run it from C:\\SignBridge_Project\\.venv."
        ) from exc

    require_file(model_path, "MediaPipe holistic_landmarker.task")
    output = np.zeros((len(image_paths), 86, 2), dtype=np.float32)
    observations: list[dict[str, Any]] = []
    frame_size = (0, 0)

    options = mp.tasks.vision.HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=min_hand_confidence,
    )

    with mp.tasks.vision.HolisticLandmarker.create_from_options(options) as landmarker:
        for index, image_path in enumerate(image_paths):
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"Could not read image: {image_path}")
            height, width = frame.shape[:2]
            frame_size = (width, height)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, index * 33)

            pose_ok = len(result.pose_landmarks) == 33
            left_ok = len(result.left_hand_landmarks) == 21
            right_ok = len(result.right_hand_landmarks) == 21

            if pose_ok:
                for joint, landmark in enumerate(result.pose_landmarks[:25]):
                    output[index, 61 + joint] = (landmark.x * width, landmark.y * height)
            if right_ok:
                for joint, landmark in enumerate(result.right_hand_landmarks):
                    output[index, joint] = (landmark.x * width, landmark.y * height)
            if left_ok:
                for joint, landmark in enumerate(result.left_hand_landmarks):
                    output[index, 21 + joint] = (landmark.x * width, landmark.y * height)

            observations.append(
                {
                    "pose": pose_ok,
                    "left_hand": left_ok,
                    "right_hand": right_ok,
                }
            )
            if (index + 1) % 25 == 0 or index + 1 == len(image_paths):
                print(f"MediaPipe JPG scan: {index + 1}/{len(image_paths)}")

    return output, observations, frame_size


def import_feature_builder():
    source_file = PROJECT_ROOT / "prepare_isharah_shared_features.py"
    require_file(source_file, "prepare_isharah_shared_features.py")
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from prepare_isharah_shared_features import build_features
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not import build_features from {source_file}: {exc}") from exc
    return build_features


def hand_swap(features: np.ndarray) -> np.ndarray:
    swapped = features.copy()
    swapped[:, 28:70] = features[:, 70:112]
    swapped[:, 70:112] = features[:, 28:70]
    swapped[:, 112:154] = features[:, 154:196]
    swapped[:, 154:196] = features[:, 112:154]
    swapped[:, 196] = features[:, 197]
    swapped[:, 197] = features[:, 196]
    return swapped


def mirror_x(features: np.ndarray) -> np.ndarray:
    mirrored = features.copy()
    spatial = mirrored[:, :SPATIAL_FEATURES].reshape(len(mirrored), -1, 2)
    spatial[:, :, 0] *= -1.0
    return mirrored


def prepare_descriptor(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Robustly scale comparable 2-D pose/hand features for DTW."""
    source = np.asarray(source[:, :SPATIAL_FEATURES], dtype=np.float32)
    target = np.asarray(target[:, :SPATIAL_FEATURES], dtype=np.float32)
    combined = np.concatenate([source, target], axis=0)
    median = np.median(combined, axis=0)
    mad = np.median(np.abs(combined - median), axis=0)
    scale = np.maximum(1.4826 * mad, 0.05)
    source_scaled = np.clip((source - median) / scale, -6.0, 6.0)
    target_scaled = np.clip((target - median) / scale, -6.0, 6.0)

    # Give articulated hands more influence than the relatively static torso.
    weights = np.ones(SPATIAL_FEATURES, dtype=np.float32)
    weights[28:196] = 1.5
    return source_scaled * weights, target_scaled * weights


def frame_cost_matrix(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    a, b = prepare_descriptor(source, target)
    diff = a[:, None, :] - b[None, :, :]
    return np.mean(diff * diff, axis=2, dtype=np.float64)


def dtw(cost: np.ndarray, band: int | None = None) -> tuple[float, list[tuple[int, int]]]:
    n, m = cost.shape
    if band is None:
        band = max(abs(n - m) + 8, int(math.ceil(max(n, m) * 0.20)))
    band = max(band, abs(n - m))

    cumulative = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    cumulative[0, 0] = 0.0
    predecessor = np.full((n, m), 255, dtype=np.uint8)

    for i in range(1, n + 1):
        expected_j = int(round(i * m / n))
        j_start = max(1, expected_j - band)
        j_end = min(m, expected_j + band)
        for j in range(j_start, j_end + 1):
            choices = (
                cumulative[i - 1, j - 1],
                cumulative[i - 1, j],
                cumulative[i, j - 1],
            )
            step = int(np.argmin(choices))
            cumulative[i, j] = cost[i - 1, j - 1] + choices[step]
            predecessor[i - 1, j - 1] = step

    if not np.isfinite(cumulative[n, m]):
        raise RuntimeError("DTW could not find a path inside the selected band")

    path: list[tuple[int, int]] = []
    i, j = n - 1, m - 1
    while i >= 0 and j >= 0:
        path.append((i, j))
        step = int(predecessor[i, j])
        if i == 0 and j == 0:
            break
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        elif step == 2:
            j -= 1
        else:
            raise RuntimeError(f"Broken DTW predecessor at ({i}, {j})")
    path.reverse()
    return float(cumulative[n, m] / len(path)), path


def map_source_index(path: list[tuple[int, int]], source_index: int) -> int:
    matches = [target for source, target in path if source == source_index]
    if not matches:
        nearest = min(path, key=lambda pair: abs(pair[0] - source_index))
        return int(nearest[1])
    return int(round(float(np.median(matches))))


def fit_fixed_coordinate_transform(
    source_points: np.ndarray,
    jpg_points: np.ndarray,
    path: list[tuple[int, int]],
) -> dict[str, float]:
    """Fit one fixed x/y scale and offset for the complete sample.

    The stored pose pickle and 256x256 JPG dump use different image-coordinate
    resolutions.  A single fixed transform is legitimate for visualization:
    it compensates only for resize/crop, never for pose or per-frame motion.
    """
    source_xy: list[np.ndarray] = []
    target_xy: list[np.ndarray] = []
    for source_index, jpg_index in path:
        a = source_points[source_index, 61 + 11 : 61 + 25]
        b = jpg_points[jpg_index, 61 + 11 : 61 + 25]
        valid = np.any(a != 0.0, axis=1) & np.any(b != 0.0, axis=1)
        if np.any(valid):
            source_xy.append(a[valid])
            target_xy.append(b[valid])
    if not source_xy:
        raise RuntimeError("No common body landmarks exist for coordinate calibration")

    source_array = np.concatenate(source_xy, axis=0).astype(np.float64)
    target_array = np.concatenate(target_xy, axis=0).astype(np.float64)

    def fit_axis(source_axis: np.ndarray, target_axis: np.ndarray) -> tuple[float, float]:
        design = np.column_stack([source_axis, np.ones(len(source_axis))])
        scale, offset = np.linalg.lstsq(design, target_axis, rcond=None)[0]
        residual = np.abs(design @ np.array([scale, offset]) - target_axis)
        cutoff = np.quantile(residual, 0.90)
        keep = residual <= cutoff
        scale, offset = np.linalg.lstsq(design[keep], target_axis[keep], rcond=None)[0]
        return float(scale), float(offset)

    scale_x, offset_x = fit_axis(source_array[:, 0], target_array[:, 0])
    scale_y, offset_y = fit_axis(source_array[:, 1], target_array[:, 1])
    return {
        "scale_x": scale_x,
        "offset_x": offset_x,
        "scale_y": scale_y,
        "offset_y": offset_y,
    }


def apply_coordinate_transform(
    points: np.ndarray, transform: dict[str, float]
) -> np.ndarray:
    transformed = points.astype(np.float32, copy=True)
    valid = np.any(transformed != 0.0, axis=-1)
    transformed[..., 0] = (
        transformed[..., 0] * transform["scale_x"] + transform["offset_x"]
    )
    transformed[..., 1] = (
        transformed[..., 1] * transform["scale_y"] + transform["offset_y"]
    )
    transformed[~valid] = 0.0
    return transformed


def calibrated_overlay_error(
    transformed_source_points: np.ndarray,
    jpg_points: np.ndarray,
    path: list[tuple[int, int]],
    frame_size: tuple[int, int],
) -> float | None:
    width, height = frame_size
    diagonal = math.hypot(width, height)
    errors: list[float] = []
    # Body landmarks 11..24 have a documented direct index correspondence.
    for source_index, jpg_index in path:
        a = transformed_source_points[source_index, 61 + 11 : 61 + 25]
        b = jpg_points[jpg_index, 61 + 11 : 61 + 25]
        valid = np.any(a != 0.0, axis=1) & np.any(b != 0.0, axis=1)
        if np.any(valid):
            errors.extend((np.linalg.norm(a[valid] - b[valid], axis=1) / diagonal).tolist())
    return float(np.median(errors)) if errors else None


def detection_rates(observations: list[dict[str, Any]]) -> dict[str, float]:
    return {
        key: round(float(np.mean([bool(row[key]) for row in observations])), 6)
        for key in ("pose", "left_hand", "right_hand")
    }


def draw_points(frame: np.ndarray, points: np.ndarray, color: tuple[int, int, int]) -> None:
    import cv2

    for x, y in points:
        if np.isfinite(x) and np.isfinite(y) and (x != 0.0 or y != 0.0):
            cv2.circle(frame, (int(round(x)), int(round(y))), 4, color, -1, cv2.LINE_AA)


def create_contact_sheet(
    output_path: Path,
    image_paths: list[Path],
    transformed_source_points: np.ndarray,
    jpg_points: np.ndarray,
    path: list[tuple[int, int]],
    clip_start: int,
    clip_end: int,
) -> None:
    import cv2

    source_indices = np.linspace(clip_start, clip_end - 1, num=6, dtype=int)
    panels: list[np.ndarray] = []
    for source_index in source_indices:
        jpg_index = map_source_index(path, int(source_index))
        frame = cv2.imread(str(image_paths[jpg_index]))
        if frame is None:
            continue
        draw_points(frame, transformed_source_points[source_index, 61:86], (0, 0, 255))
        draw_points(frame, jpg_points[jpg_index, 61:86], (0, 255, 0))
        draw_points(frame, transformed_source_points[source_index, 0:42], (255, 0, 255))
        draw_points(frame, jpg_points[jpg_index, 0:42], (0, 255, 255))
        cv2.putText(
            frame,
            f"pose {source_index} -> JPG {jpg_index}",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        target_width = 420
        scale = target_width / frame.shape[1]
        panel = cv2.resize(frame, (target_width, int(round(frame.shape[0] * scale))))
        panels.append(panel)

    if not panels:
        raise RuntimeError("Could not create any contact-sheet panels")
    max_height = max(panel.shape[0] for panel in panels)
    padded = []
    for panel in panels:
        canvas = np.zeros((max_height, panel.shape[1], 3), dtype=np.uint8)
        canvas[: panel.shape[0]] = panel
        padded.append(canvas)
    rows = [np.hstack(padded[index : index + 3]) for index in range(0, len(padded), 3)]
    sheet = np.vstack(rows)
    cv2.putText(
        sheet,
        "RED/MAGENTA=stored pose | GREEN/YELLOW=fresh MediaPipe",
        (20, sheet.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"Could not write contact sheet: {output_path}")


def run_self_test() -> int:
    rng = np.random.default_rng(42)
    source = rng.normal(size=(30, 12)).astype(np.float32)
    target_indices = np.linspace(0, 29, 36).round().astype(int)
    target = source[target_indices] + rng.normal(scale=0.01, size=(36, 12))
    cost = np.mean((source[:, None, :] - target[None, :, :]) ** 2, axis=2)
    score, path = dtw(cost, band=12)
    mapped = [map_source_index(path, i) for i in range(30)]
    if score >= 0.01 or mapped != sorted(mapped):
        raise AssertionError("DTW self-test failed")
    print("Self-test passed")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()

    pose_dir = args.pose_dir.resolve()
    clips_csv = args.clips_csv.resolve()
    frames_dir = (args.frames_root / args.sample_id).resolve()
    model_path = args.model.resolve()
    output_dir = (args.output_root / f"isharah_alignment_{args.sample_id}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_points, sample_row = load_source_sequence(pose_dir, args.sample_id)
    clip_row = find_clip_row(clips_csv, args.sample_id, args.gloss)
    image_paths = list_images(frames_dir)
    clip_start = int(clip_row["relative_start_frame"])
    clip_end = int(clip_row["relative_end_frame"])
    if not (0 <= clip_start < clip_end <= len(source_points)):
        raise ValueError(
            f"Invalid clip [{clip_start}, {clip_end}) for {len(source_points)} pose frames"
        )

    print("Isharah alignment diagnostic")
    print(f"Sample: {args.sample_id} | gloss: {args.gloss}")
    print(f"Stored pose frames: {len(source_points)}")
    print(f"JPG frames: {len(image_paths)}")
    print(f"Pose-timeline gloss interval: [{clip_start}, {clip_end})")

    jpg_points, observations, frame_size = extract_jpg_keypoints(
        image_paths, model_path, args.min_hand_confidence
    )
    build_features = import_feature_builder()
    source_features = build_features(source_points)
    jpg_features = build_features(jpg_points)

    variants = {
        "direct": source_features,
        "hands_swapped": hand_swap(source_features),
        "mirrored_x": mirror_x(source_features),
        "hands_swapped_and_mirrored_x": mirror_x(hand_swap(source_features)),
    }
    candidates: list[dict[str, Any]] = []
    paths: dict[str, list[tuple[int, int]]] = {}
    for name, candidate in variants.items():
        score, alignment_path = dtw(frame_cost_matrix(candidate, jpg_features))
        reverse_score, _ = dtw(frame_cost_matrix(candidate, jpg_features[::-1]))
        candidates.append(
            {
                "variant": name,
                "dtw_cost": score,
                "reversed_control_cost": reverse_score,
                "control_separation_ratio": reverse_score / max(score, 1e-12),
            }
        )
        paths[name] = alignment_path
    candidates.sort(key=lambda row: row["dtw_cost"])
    best = candidates[0]
    runner_up = candidates[1]
    best_path = paths[best["variant"]]

    jpg_start = map_source_index(best_path, clip_start)
    jpg_end_inclusive = map_source_index(best_path, clip_end - 1)
    mapped_end = min(len(image_paths), jpg_end_inclusive + 1)
    coordinate_transform = fit_fixed_coordinate_transform(
        source_points, jpg_points, best_path
    )
    transformed_source_points = apply_coordinate_transform(
        source_points, coordinate_transform
    )
    overlay_error = calibrated_overlay_error(
        transformed_source_points, jpg_points, best_path, frame_size
    )
    variant_margin = runner_up["dtw_cost"] / max(best["dtw_cost"], 1e-12)
    control_ratio = best["control_separation_ratio"]

    if control_ratio >= 1.35 and variant_margin >= 1.05:
        verdict = "PROMISING_REVIEW_PREVIEW"
    elif control_ratio >= 1.15:
        verdict = "AMBIGUOUS_REVIEW_REQUIRED"
    else:
        verdict = "FAILED_NO_RELIABLE_ALIGNMENT"

    preview_path = output_dir / "alignment_preview.jpg"
    create_contact_sheet(
        preview_path,
        image_paths,
        transformed_source_points,
        jpg_points,
        best_path,
        clip_start,
        clip_end,
    )

    report = {
        "schema": "signbridge-isharah-alignment-diagnostic-v1",
        "read_only_diagnostic": True,
        "sample_id": args.sample_id,
        "gloss": args.gloss,
        "sample_row": sample_row,
        "clip": {
            "clip_id": clip_row.get("clip_id"),
            "pose_start_inclusive": clip_start,
            "pose_end_exclusive": clip_end,
            "pose_clip_frames": clip_end - clip_start,
            "mapped_jpg_start_inclusive": jpg_start,
            "mapped_jpg_end_exclusive": mapped_end,
            "mapped_jpg_clip_frames": mapped_end - jpg_start,
            "mapped_first_filename": image_paths[jpg_start].name,
            "mapped_last_filename": image_paths[mapped_end - 1].name,
        },
        "counts": {
            "stored_pose_frames": len(source_points),
            "jpg_frames": len(image_paths),
            "difference_jpg_minus_pose": len(image_paths) - len(source_points),
        },
        "frame_size": {"width": frame_size[0], "height": frame_size[1]},
        "fresh_mediapipe_detection_rates": detection_rates(observations),
        "candidate_variants": candidates,
        "best_variant": best["variant"],
        "best_vs_runner_up_ratio": variant_margin,
        "fixed_coordinate_transform_source_to_jpg": coordinate_transform,
        "median_calibrated_body_overlay_error_fraction_of_image_diagonal": overlay_error,
        "verdict": verdict,
        "important": (
            "This result is diagnostic only. Do not unlock batch generation until "
            "the preview is visually reviewed and several samples pass consistently."
        ),
        "outputs": {"preview": str(preview_path)},
    }
    report_path = output_dir / "alignment_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print()
    print("RESULT")
    print(f"Verdict: {verdict}")
    print(f"Best variant: {best['variant']}")
    print(f"DTW cost: {best['dtw_cost']:.6f}")
    print(f"Reversed-control separation: {control_ratio:.3f}x")
    print(f"Best-vs-runner-up separation: {variant_margin:.3f}x")
    print(f"Mapped JPG interval: [{jpg_start}, {mapped_end})")
    print(f"Report: {report_path}")
    print(f"Preview: {preview_path}")
    print("No manifest, model, dataset, motion file, or avatar code was changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
