"""Convert the bundled LabelMe polygon annotations to COCO JSON files.

With no arguments, this script reads:

    image datasets/manual annotations/training/*.json
    image datasets/manual annotations/validation/*.json

and writes:

    image datasets/annotation COCO dataset/instances_train2019.json
    image datasets/annotation COCO dataset/instances_val2019.json
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


CATEGORIES = [
    {"id": 1, "name": "single layer", "supercategory": "material"},
    {"id": 2, "name": "contaminants", "supercategory": "material"},
    {"id": 3, "name": "multilayer", "supercategory": "material"},
]
CATEGORY_IDS = {category["name"]: category["id"] for category in CATEGORIES}
SPLITS = {
    "training": ("training dataset", "instances_train2019.json"),
    "validation": ("validation dataset", "instances_val2019.json"),
}


def normalize_label(label: str) -> str:
    """Normalize harmless LabelMe spelling differences before category lookup."""
    return " ".join(label.strip().lower().replace("_", " ").split())


def polygon_area(points: Sequence[Tuple[float, float]]) -> float:
    """Return the absolute area of a polygon using the shoelace formula."""
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def parse_points(
    raw_points: Iterable[Sequence[float]],
    width: int,
    height: int,
    source: Path,
    shape_index: int,
) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for point_index, point in enumerate(raw_points):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(
                f"{source}: shape {shape_index}, point {point_index} is not an [x, y] pair"
            )
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(
                f"{source}: shape {shape_index}, point {point_index} is not finite"
            )
        if x < 0 or x > width or y < 0 or y > height:
            raise ValueError(
                f"{source}: shape {shape_index}, point {point_index} is outside the image"
            )
        points.append((x, y))

    if len(points) < 3:
        raise ValueError(f"{source}: shape {shape_index} has fewer than three points")
    return points


def load_labelme(path: Path) -> Dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read LabelMe JSON {path}: {exc}") from exc

    required = ("imagePath", "imageWidth", "imageHeight", "shapes")
    missing = [field for field in required if field not in data]
    if missing:
        raise ValueError(f"{path}: missing LabelMe fields: {', '.join(missing)}")
    return data


def convert_split(annotation_dir: Path, image_dir: Path, output_path: Path) -> None:
    if not annotation_dir.is_dir():
        raise FileNotFoundError(f"LabelMe annotation directory does not exist: {annotation_dir}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Training image directory does not exist: {image_dir}")

    labelme_files = sorted(
        path
        for path in annotation_dir.glob("*.json")
        if not path.name.lower().startswith(("instances", "._"))
    )
    if not labelme_files:
        raise RuntimeError(f"No LabelMe JSON files found in: {annotation_dir}")

    coco = {"images": [], "categories": CATEGORIES, "annotations": []}
    seen_image_names = set()
    annotation_id = 1

    for image_id, labelme_path in enumerate(labelme_files, start=1):
        data = load_labelme(labelme_path)
        width = int(data["imageWidth"])
        height = int(data["imageHeight"])
        if width <= 0 or height <= 0:
            raise ValueError(f"{labelme_path}: image dimensions must be positive")

        image_name = str(data["imagePath"]).replace("\\", "/").rsplit("/", 1)[-1]
        if not image_name:
            raise ValueError(f"{labelme_path}: imagePath is empty")
        if image_name in seen_image_names:
            raise ValueError(f"{labelme_path}: duplicate imagePath: {image_name}")
        if not (image_dir / image_name).is_file():
            raise FileNotFoundError(
                f"{labelme_path}: referenced training image does not exist: {image_dir / image_name}"
            )
        seen_image_names.add(image_name)

        coco["images"].append(
            {"height": height, "width": width, "id": image_id, "file_name": image_name}
        )

        shapes = data["shapes"]
        if not isinstance(shapes, list):
            raise ValueError(f"{labelme_path}: shapes must be a list")

        for shape_index, shape in enumerate(shapes, start=1):
            shape_type = shape.get("shape_type", "polygon")
            if shape_type != "polygon":
                raise ValueError(
                    f"{labelme_path}: shape {shape_index} has unsupported type {shape_type!r}"
                )

            label = normalize_label(str(shape.get("label", "")))
            if label not in CATEGORY_IDS:
                expected = ", ".join(CATEGORY_IDS)
                raise ValueError(
                    f"{labelme_path}: shape {shape_index} has unknown label {label!r}; "
                    f"expected one of: {expected}"
                )

            points = parse_points(
                shape.get("points", []), width, height, labelme_path, shape_index
            )
            area = polygon_area(points)
            if area <= 0:
                raise ValueError(f"{labelme_path}: shape {shape_index} has zero area")

            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            segmentation = [coordinate for point in points for coordinate in point]

            coco["annotations"].append(
                {
                    "segmentation": [segmentation],
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "area": area,
                    "category_id": CATEGORY_IDS[label],
                    "id": annotation_id,
                }
            )
            annotation_id += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(coco, handle, ensure_ascii=False, indent=4)
        handle.write("\n")
    os.replace(temporary_path, output_path)

    print(
        f"[{annotation_dir.name}] {len(coco['images'])} images, "
        f"{len(coco['annotations'])} annotations -> {output_path}"
    )


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repository_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Convert LabelMe training and validation polygons to COCO JSON"
    )
    parser.add_argument(
        "--annotations_root",
        type=Path,
        default=repository_root / "image datasets" / "manual annotations",
        help="Directory containing training/ and validation/ LabelMe JSON folders",
    )
    parser.add_argument(
        "--image_dataset_root",
        type=Path,
        default=repository_root / "image datasets",
        help="Directory containing training dataset/ and validation dataset/ images",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=repository_root / "image datasets" / "annotation COCO dataset",
        help="Destination directory for both COCO JSON files",
    )
    args = parser.parse_args()

    for split_name, (image_folder, output_name) in SPLITS.items():
        convert_split(
            annotation_dir=args.annotations_root / split_name,
            image_dir=args.image_dataset_root / image_folder,
            output_path=args.output_dir / output_name,
        )


if __name__ == "__main__":
    main()
