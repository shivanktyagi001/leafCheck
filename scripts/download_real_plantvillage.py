"""
Download real PlantVillage tomato leaf images from GitHub repository.
Downloads authentic photos for all 10 tomato disease classes.
"""

import os
import urllib.request
import urllib.parse
import json
import logging
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("download_plantvillage")

TOMATO_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

RAW_DIR = "dataset/raw"
SAMPLES_PER_CLASS = 25


def download_file(url: str, dest_path: str):
    try:
        urllib.request.urlretrieve(url, dest_path)
    except Exception as e:
        logger.warning(f"Failed to download {url}: {e}")


def fetch_class_images(cls_name: str):
    cls_dir = os.path.join(RAW_DIR, cls_name)
    os.makedirs(cls_dir, exist_ok=True)

    # Clean out synthetic samples
    for f in os.listdir(cls_dir):
        if f.startswith("sample_"):
            os.remove(os.path.join(cls_dir, f))

    encoded_cls = urllib.parse.quote(cls_name)
    api_url = f"https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color/{encoded_cls}"

    req = urllib.request.Request(api_url, headers={"User-Agent": "LeafCheck-Setup/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as res:
            items = json.loads(res.read().decode("utf-8"))
    except Exception as e:
        logger.error(f"Error fetching directory list for {cls_name}: {e}")
        return

    image_items = [i for i in items if i["name"].lower().endswith((".jpg", ".jpeg", ".png"))][:SAMPLES_PER_CLASS]
    logger.info(f"Downloading {len(image_items)} real images for {cls_name}...")

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for item in image_items:
            download_url = item["download_url"]
            dest = os.path.join(cls_dir, item["name"])
            futures.append(executor.submit(download_file, download_url, dest))
        for f in futures:
            f.result()


def main():
    logger.info("Starting download of authentic PlantVillage tomato leaf images...")
    for cls in TOMATO_CLASSES:
        fetch_class_images(cls)
    logger.info("Download complete. Re-splitting dataset...")


if __name__ == "__main__":
    main()
