"""Extract an image-derived surface degradation / spallation proxy.

This is deliberately a transparent classical-vision score rather than a
ground-truth label. It measures dispersed dark/coloured material and fragment
complexity inside each crucible. The final score is a within-time percentile
rank, aggregated across available times with later times weighted more heavily.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from detect_cups import (
    ROOT,
    detect_circles,
    discover_time_images,
    index_circles,
    read_image,
    write_image,
)


OUTPUT_DIR = ROOT / "analysis_outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def normalized_crop(
    image: np.ndarray, x: float, y: float, radius: float, size: int = 320
) -> np.ndarray:
    half = int(round(radius))
    center = (int(round(x)), int(round(y)))
    padded = cv2.copyMakeBorder(
        image, half, half, half, half, cv2.BORDER_REFLECT_101
    )
    crop = cv2.getRectSubPix(padded, (2 * half, 2 * half), (center[0] + half, center[1] + half))
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def extract_features(crop: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    size = crop.shape[0]
    yy, xx = np.ogrid[:size, :size]
    center = (size - 1) / 2
    radial = np.sqrt((xx - center) ** 2 + (yy - center) ** 2) / (size / 2)
    interior = radial <= 0.66

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    saturation = hsv[..., 1].astype(float) / 255.0
    value = hsv[..., 2].astype(float) / 255.0
    values = value[interior]
    low, high = np.quantile(values, [0.08, 0.92])
    darkness = np.clip((high - value) / max(high - low, 1e-6), 0, 1)
    material_score = np.maximum(darkness, 0.85 * saturation)
    binary = ((material_score >= 0.42) & interior).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    interior_area = float(interior.sum())
    min_area = max(12, int(0.0008 * interior_area))
    valid = [
        idx
        for idx in range(1, count)
        if stats[idx, cv2.CC_STAT_AREA] >= min_area
    ]
    valid_areas = np.array([stats[idx, cv2.CC_STAT_AREA] for idx in valid], dtype=float)
    if len(valid):
        largest_label = valid[int(np.argmax(valid_areas))]
        largest_area = float(stats[largest_label, cv2.CC_STAT_AREA])
        debris_labels = [idx for idx in valid if idx != largest_label]
        debris_mask = np.isin(labels, debris_labels)
        largest_centroid = centroids[largest_label]
    else:
        largest_label = -1
        largest_area = 0.0
        debris_labels = []
        debris_mask = np.zeros_like(binary, dtype=bool)
        largest_centroid = np.array([center, center])

    material_area = float(binary[interior].sum())
    debris_area = float(debris_mask[interior].sum())
    if debris_area:
        debris_y, debris_x = np.where(debris_mask)
        debris_distance = np.sqrt(
            (debris_x - largest_centroid[0]) ** 2
            + (debris_y - largest_centroid[1]) ** 2
        )
        debris_dispersion = float(np.mean(debris_distance) / (size / 2))
    else:
        debris_dispersion = 0.0

    edges = cv2.Canny(gray, 40, 120)
    edge_fraction = float((edges.astype(bool) & interior).sum() / interior_area)
    blue_mask = (
        (hsv[..., 0] >= 85)
        & (hsv[..., 0] <= 135)
        & (saturation >= 0.20)
        & interior
    )
    annulus = interior & (radial >= 0.28)
    material_outside_center = float(binary[annulus].sum() / max(annulus.sum(), 1))
    fragmentation_ratio = debris_area / max(material_area, 1.0)

    overlay = crop.copy()
    overlay[debris_mask] = (
        0.25 * overlay[debris_mask] + 0.75 * np.array([0, 140, 255])
    ).astype(np.uint8)
    if largest_label >= 0:
        largest_contours, _ = cv2.findContours(
            (labels == largest_label).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, largest_contours, -1, (0, 190, 0), 2)
    cv2.circle(overlay, (round(center), round(center)), round(0.66 * size / 2), (230, 230, 230), 1)

    features = {
        "material_fraction": material_area / interior_area,
        "largest_component_fraction": largest_area / interior_area,
        "debris_fraction": debris_area / interior_area,
        "fragmentation_ratio": fragmentation_ratio,
        "debris_component_count": float(len(debris_labels)),
        "debris_dispersion": debris_dispersion,
        "material_outside_center_fraction": material_outside_center,
        "edge_fraction": edge_fraction,
        "blue_fraction": float(blue_mask.sum() / interior_area),
        "local_value_p08": float(low),
        "local_value_p92": float(high),
    }
    return features, overlay


def make_contact_sheet(
    crops: dict[int, np.ndarray], time_hours: int, scores: pd.Series | None = None
) -> None:
    tile = 240
    sheet = np.full((4 * tile, 8 * tile, 3), 245, dtype=np.uint8)
    for sample_id in range(1, 33):
        if sample_id not in crops:
            continue
        row = (sample_id - 1) // 8
        col = (sample_id - 1) % 8
        crop = cv2.resize(crops[sample_id], (tile, tile), interpolation=cv2.INTER_AREA)
        x0, y0 = col * tile, row * tile
        sheet[y0 : y0 + tile, x0 : x0 + tile] = crop
        label = f"#{sample_id}"
        if scores is not None:
            label += f"  {scores.loc[sample_id]:.2f}"
        cv2.rectangle(sheet, (x0, y0), (x0 + tile, y0 + 28), (255, 255, 255), -1)
        cv2.putText(
            sheet,
            label,
            (x0 + 8, y0 + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )
    write_image(FIGURE_DIR / f"macro_{time_hours}h_contact_sheet.jpg", sheet, quality=94)


def percentile_score(frame: pd.DataFrame) -> pd.Series:
    fields = {
        "debris_fraction": 0.30,
        "fragmentation_ratio": 0.20,
        "debris_component_count": 0.15,
        "debris_dispersion": 0.15,
        "material_outside_center_fraction": 0.10,
        "edge_fraction": 0.10,
    }
    score = pd.Series(0.0, index=frame.index)
    for field, weight in fields.items():
        score += weight * frame[field].rank(method="average", pct=True)
    return score


def summarize_features(features: pd.DataFrame) -> pd.DataFrame:
    """Weight available times by chronological rank, normalized per sample."""
    times = sorted(features["time_h"].unique())
    weights = {time: rank for rank, time in enumerate(times, start=1)}
    weighted = features.assign(time_weight=features["time_h"].map(weights))
    weighted["weighted_score"] = weighted["time_rank_score"] * weighted["time_weight"]
    aggregate = weighted.groupby("sample_id", as_index=False).agg(
        spallation_proxy=("weighted_score", "sum"),
        observed_weight=("time_weight", "sum"),
    )
    aggregate["spallation_proxy"] /= aggregate.pop("observed_weight")
    aggregate["spallation_proxy"] = aggregate["spallation_proxy"].rank(pct=True)
    latest = features.sort_values("time_h").groupby("sample_id").tail(1).set_index("sample_id")
    aggregate["latest_time_h"] = aggregate["sample_id"].map(latest["time_h"])
    aggregate["spallation_latest_proxy"] = aggregate["sample_id"].map(latest["time_rank_score"])
    hundred = features.loc[features["time_h"] == 100].set_index("sample_id")["time_rank_score"]
    aggregate["spallation_100h_proxy"] = aggregate["sample_id"].map(hundred)
    return aggregate


def main(input_dir: Path = ROOT) -> None:
    image_paths = discover_time_images(input_dir)
    rows: list[dict[str, float | int]] = []
    stored_crops: dict[int, dict[int, np.ndarray]] = {}
    for time_hours in image_paths:
        image = read_image(image_paths[time_hours])
        circles = detect_circles(image)
        indexed, layout = index_circles(circles, image.shape)
        crops: dict[int, np.ndarray] = {}
        for sample_id, x, y, radius in indexed:
            crop = normalized_crop(image, x, y, radius)
            features, overlay = extract_features(crop)
            rows.append({"sample_id": sample_id, "time_h": time_hours, **features})
            crops[sample_id] = overlay
        stored_crops[time_hours] = crops
        print(f"{time_hours} h: {layout}, extracted {len(crops)} cups")

    features = pd.DataFrame(rows).sort_values(["sample_id", "time_h"])
    features["time_rank_score"] = pd.Series(index=features.index, dtype=float)
    for _, frame in features.groupby("time_h"):
        features.loc[frame.index, "time_rank_score"] = percentile_score(frame)
    aggregate = summarize_features(features)

    features.to_csv(OUTPUT_DIR / "macro_image_features.csv", index=False)
    aggregate.to_csv(OUTPUT_DIR / "spallation_proxy.csv", index=False)
    for time_hours, crops in stored_crops.items():
        time_scores = features.loc[
            features["time_h"] == time_hours
        ].set_index("sample_id")["time_rank_score"]
        make_contact_sheet(crops, time_hours, time_scores)
    print(aggregate.sort_values("spallation_proxy").to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=ROOT,
        help="Folder containing time images such as 5h.jpg (default: script folder).",
    )
    args = parser.parse_args()
    main(args.input_dir)
