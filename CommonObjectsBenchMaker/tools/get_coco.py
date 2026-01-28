#!/usr/bin/env python3
"""
get_coco.py

Download images from the COCO dataset using FiftyOne for the CommonObjectsBench benchmark.
Supports configurable class filtering and random sampling.
"""
import os
import shutil
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import fiftyone.zoo as foz

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
# COCO class names (all 80 classes by default)
# To use specific classes, modify this list or set to None to use all classes
#TODO: Set these to the COCO classes you want to use
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# Total sample size (if set, will sample this many images total across all classes)
# Set to None to use all available images
#TODO: Set these to the number of images you want to use
MAX_SAMPLES = None  # Set to a number to limit total samples (e.g., 1000)

# Output configuration
OUTPUT_DIR = "/tmp/CommonObjectsBenchMaker/images/coco"
RANDOM_SEED = 42
MAX_WORKERS = 10

# COCO dataset split to use
COCO_SPLIT = "train"  # "train" or "val"

def copy_image(src_path, dst_path):
    """Copy a single image file."""
    try:
        if os.path.exists(dst_path):
            return {"status": "skipped", "filepath": str(dst_path)}
        
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)
        return {"status": "success", "filepath": str(dst_path)}
    except Exception as e:
        logger.error(f"Failed to copy {src_path}: {e}")
        return {"status": "error", "filepath": str(dst_path), "error": str(e)}

def get_coco_images(classes, split=COCO_SPLIT, max_samples=None, random_seed=RANDOM_SEED):
    """Load COCO dataset and filter by classes using FiftyOne's built-in filtering."""
    random.seed(random_seed)
    logger.info(f"Using random seed: {random_seed}")
    
    logger.info(f"Loading COCO {split} dataset with classes: {classes}")
    logger.info(f"Max samples: {max_samples if max_samples else 'all'}")
    
    try:
        # Load COCO dataset using FiftyOne with built-in class filtering
        dataset = foz.load_zoo_dataset(
            "coco-2017",
            split=split,
            label_types=["detections", "captions"],
            classes=classes,
            max_samples=max_samples,
        )
        logger.info(f"Loaded COCO dataset with {len(dataset)} samples")
        return list(dataset)
    except Exception as e:
        logger.error(f"Failed to load COCO dataset: {e}")
        raise

def download_coco_images(samples, output_dir, max_workers=MAX_WORKERS):
    """Download/copy COCO images to output directory."""
    # Prepare copy tasks
    tasks = []
    for sample in samples:
        src_path = sample.filepath
        # Create relative path structure: coco/train2017/filename.jpg
        filename = os.path.basename(src_path)
        # Extract split name from path (e.g., train2017, val2017)
        split_name = os.path.basename(os.path.dirname(src_path))
        dst_path = Path(output_dir) / split_name / filename
        tasks.append((src_path, dst_path))
    
    logger.info(f"Starting copy of {len(tasks)} images to {output_dir}")
    
    # Copy in parallel
    results = {"success": 0, "skipped": 0, "error": 0}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(copy_image, src, dst)
            for src, dst in tasks
        ]
        
        for future in as_completed(futures):
            result = future.result()
            results[result["status"]] += 1
    
    logger.info(f"Completed: {results['success']} success, "
                f"{results['skipped']} skipped, {results['error']} errors")
    return results

def main():
    """Main function to download COCO images."""
    logger.info("Starting COCO image download...")
    logger.info(f"Target classes: {COCO_CLASSES}")
    logger.info(f"Max samples: {MAX_SAMPLES if MAX_SAMPLES else 'all'}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    try:
        # Get filtered and sampled images using FiftyOne's built-in filtering
        samples = get_coco_images(
            classes=COCO_CLASSES,
            split=COCO_SPLIT,
            max_samples=MAX_SAMPLES,
            random_seed=RANDOM_SEED
        )
        
        # Download/copy images
        download_coco_images(samples, OUTPUT_DIR, max_workers=MAX_WORKERS)
        
        logger.info("All downloads completed!")
        
    except Exception as e:
        logger.error(f"Failed to download COCO images: {e}")
        raise

if __name__ == "__main__":
    main()

