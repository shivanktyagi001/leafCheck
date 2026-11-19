"""
LeafCheck - Exploratory Data Analysis (EDA) Module
Analyzes class distributions, image dimensions, corruptions, and generates
visual analytics for dataset validation.
"""

import os
import glob
import logging
from typing import Dict, List, Tuple, Any

# Keep matplotlib cache within project workspace
os.environ["MPLCONFIGDIR"] = os.path.abspath(".matplotlib")

from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("leafcheck.eda")


def analyze_dataset_directory(base_dir: str = "dataset") -> Dict[str, Any]:
    """
    Scans dataset splits (train, validation, test) and computes detailed statistics.
    """
    splits = ["train", "validation", "test"]
    stats = {}
    records = []

    for split in splits:
        split_path = os.path.join(base_dir, split)
        if not os.path.exists(split_path):
            continue

        class_dirs = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
        split_count = 0

        for cls in sorted(class_dirs):
            cls_folder = os.path.join(split_path, cls)
            valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
            image_files = []
            for ext in valid_exts:
                image_files.extend(glob.glob(os.path.join(cls_folder, ext)))

            count = len(image_files)
            split_count += count
            records.append({
                "split": split,
                "class_name": cls,
                "image_count": count
            })

        stats[split] = split_count

    df = pd.DataFrame(records)
    return {"df": df, "split_totals": stats}


def plot_class_distribution(df: pd.DataFrame, output_path: str = "monitoring/class_distribution.png"):
    """Generates and saves a stacked or grouped bar plot of class distribution."""
    if df.empty:
        logger.warning("Empty dataframe, skipping distribution plot.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pivot_df = df.pivot(index="class_name", columns="split", values="image_count").fillna(0)

    fig, ax = plt.subplots(figsize=(12, 7))
    pivot_df.plot(kind="bar", stacked=True, ax=ax, colormap="viridis", edgecolor="black")

    ax.set_title("LeafCheck - Tomato Disease Class Distribution Across Splits", fontsize=14, fontweight="bold")
    ax.set_xlabel("Tomato Disease Class", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Class distribution chart saved to {output_path}")


def analyze_image_dimensions(sample_paths: List[str]) -> pd.DataFrame:
    """Analyzes dimensions, aspect ratios, and color modes of images."""
    data = []
    for p in sample_paths:
        try:
            with Image.open(p) as img:
                w, h = img.size
                data.append({
                    "path": p,
                    "width": w,
                    "height": h,
                    "aspect_ratio": round(w / h, 2),
                    "mode": img.mode
                })
        except Exception as e:
            logger.warning(f"Failed to read image {p}: {e}")

    return pd.DataFrame(data)


if __name__ == "__main__":
    result = analyze_dataset_directory("dataset")
    df = result["df"]
    if not df.empty:
        print(df.to_string())
        plot_class_distribution(df)
    else:
        print("Dataset directory is empty or not yet prepared.")
