"""Detect and index the 32 crucibles in each macroscopic corrosion image.

The two trays are interpreted row-major: samples 1-16 on the left and
17-32 on the right, matching the experiment description.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_outputs" / "cup_detection"
OUT.mkdir(parents=True, exist_ok=True)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def discover_time_images(directory: Path = ROOT) -> dict[int, Path]:
    """Find existing <hours>h images in numeric time order (one per time)."""
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Input directory does not exist or is not a folder: {directory}")
    images: dict[int, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        match = re.fullmatch(r"(\d+)h", path.stem, flags=re.IGNORECASE)
        if match is None:
            continue
        time = int(match.group(1))
        previous = images.get(time)
        if previous is None or IMAGE_EXTENSIONS.index(path.suffix.lower()) < IMAGE_EXTENSIONS.index(previous.suffix.lower()):
            images[time] = path
    if not images:
        raise FileNotFoundError(
            f"No time images found in {directory}. Use names such as 0h.jpg, 5h.tif or 100h.png."
        )
    return dict(sorted(images.items()))


def find_time_image(time_hours: int, directory: Path = ROOT) -> Path:
    """Find <time>h images case-insensitively, preferring IMAGE_EXTENSIONS order."""
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Input directory does not exist or is not a folder: {directory}")
    candidates = {
        path.name.casefold(): path
        for path in sorted(directory.iterdir())
        if path.is_file()
    }
    for extension in IMAGE_EXTENSIONS:
        name = f"{time_hours}h{extension}"
        if name in candidates:
            return candidates[name]
    formats = ", ".join(IMAGE_EXTENSIONS)
    raise FileNotFoundError(
        f"No image found for {time_hours}h in {directory}. Supported formats: {formats}"
    )


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def write_image(path: Path, image: np.ndarray, quality: int = 92) -> None:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    encoded.tofile(path)


def detect_circles(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = 1800.0 / max(height, width)
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
    min_dim = min(small.shape[:2])
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dim / 11,
        param1=110,
        param2=42,
        minRadius=int(min_dim / 35),
        maxRadius=int(min_dim / 13),
    )
    if circles is None:
        raise RuntimeError("No crucible circles detected")
    return circles[0] / scale


def one_dimensional_centers(values: np.ndarray) -> np.ndarray:
    model = KMeans(n_clusters=4, random_state=42, n_init=30)
    model.fit(values.reshape(-1, 1))
    return np.sort(model.cluster_centers_.ravel())


def grid_from_candidates(
    candidates: np.ndarray, sample_offset: int
) -> list[tuple[int, float, float, float]]:
    """Fit a robust 4x4 grid and use Hough circles nearest each grid point."""
    if len(candidates) < 12:
        raise RuntimeError(f"Only {len(candidates)} candidate cups for a 4x4 tray")
    x_centers = one_dimensional_centers(candidates[:, 0])
    y_centers = one_dimensional_centers(candidates[:, 1])
    x_spacing = float(np.median(np.diff(x_centers)))
    y_spacing = float(np.median(np.diff(y_centers)))
    radius_median = float(np.median(candidates[:, 2]))
    available = set(range(len(candidates)))
    indexed: list[tuple[int, float, float, float]] = []
    local_index = 0
    for target_y in y_centers:
        for target_x in x_centers:
            local_index += 1
            if available:
                choices = np.array(sorted(available), dtype=int)
                trial = candidates[choices]
                cost = (
                    ((trial[:, 0] - target_x) / x_spacing) ** 2
                    + ((trial[:, 1] - target_y) / y_spacing) ** 2
                    + 0.1 * ((trial[:, 2] - radius_median) / radius_median) ** 2
                )
                best_position = int(np.argmin(cost))
                best_index = int(choices[best_position])
                if float(cost[best_position]) <= 0.40:
                    circle = candidates[best_index].copy()
                    circle[2] = radius_median
                    available.remove(best_index)
                else:
                    circle = np.array([target_x, target_y, radius_median])
            else:
                circle = np.array([target_x, target_y, radius_median])
            indexed.append((sample_offset + local_index, *map(float, circle)))
    return indexed


def index_circles(
    circles: np.ndarray, image_shape: tuple[int, int, int]
) -> tuple[list[tuple[int, float, float, float]], str]:
    """Index two horizontal trays, with a fallback for a single tray."""
    _, width = image_shape[:2]
    left = circles[circles[:, 0] < 0.48 * width]
    right = circles[circles[:, 0] > 0.52 * width]
    if len(left) >= 12 and len(right) >= 12:
        indexed = grid_from_candidates(left, 0) + grid_from_candidates(right, 16)
        return indexed, "two_trays_1_to_32"
    # The supplied 50 h image contains only the visually matching right tray.
    indexed = grid_from_candidates(circles, 16)
    return indexed, "single_tray_assumed_17_to_32"


def main(input_dir: Path = ROOT) -> None:
    image_paths = discover_time_images(input_dir)
    for image_path in image_paths.values():
        image = read_image(image_path)
        thumbnail_scale = 1200 / max(image.shape[:2])
        thumbnail = cv2.resize(
            image, None, fx=thumbnail_scale, fy=thumbnail_scale, interpolation=cv2.INTER_AREA
        )
        write_image(OUT / f"{image_path.stem}_thumbnail.jpg", thumbnail)
        circles = detect_circles(image)
        print(image_path.name, "raw circles:", len(circles))
        indexed, layout = index_circles(circles, image.shape)
        print("layout:", layout)
        overlay = image.copy()
        for sample_id, x, y, radius in indexed:
            cv2.circle(overlay, (round(x), round(y)), round(radius), (0, 180, 0), 8)
            cv2.putText(
                overlay,
                str(sample_id),
                (round(x - 0.35 * radius), round(y + 0.15 * radius)),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(1.0, radius / 100),
                (0, 0, 255),
                max(3, round(radius / 35)),
                cv2.LINE_AA,
            )
        preview_scale = 1800 / max(overlay.shape[:2])
        preview = cv2.resize(
            overlay, None, fx=preview_scale, fy=preview_scale, interpolation=cv2.INTER_AREA
        )
        output_path = OUT / f"{image_path.stem}_detected.jpg"
        write_image(output_path, preview)
        print("wrote", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=ROOT,
        help="Folder containing time images such as 5h.jpg (default: script folder).",
    )
    args = parser.parse_args()
    main(args.input_dir)
