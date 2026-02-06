#!/usr/bin/env python3
"""
ccsn_meta.py

Create metadata.jsonl file for CCSN images with image_id and cloud_category.
The image_id is the relative path from IMAGE_ROOT_DIR.
"""
import os
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
IMAGE_ROOT_DIR = "/tmp/CloudBench/images"
CCSN_DIR = os.path.join(IMAGE_ROOT_DIR, "ccsn")
METADATA_JSONL = os.path.join(IMAGE_ROOT_DIR, "metadata.jsonl")

# Cloud category mapping: folder name -> full category name
CATEGORY_MAPPING = {
    "Ci": "cirrus",
    "Cs": "cirrostratus",
    "Cc": "cirrocumulus",
    "Ac": "altocumulus",
    "As": "altostratus",
    "Cu": "cumulus",
    "Cb": "cumulonimbus",
    "Ns": "nimbostratus",
    "Sc": "stratocumulus",
    "St": "stratus",
    "Ct": "contrail"
}

# Image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff', '.webp'}

def get_cloud_category(folder_name):
    """
    Map folder name to cloud category.
    
    Args:
        folder_name: Name of the folder containing images
    
    Returns:
        Full cloud category name, or None if not found
    """
    # Try exact match first
    if folder_name in CATEGORY_MAPPING:
        return CATEGORY_MAPPING[folder_name]
    
    # Try case-insensitive match
    folder_name_lower = folder_name.lower()
    for key, value in CATEGORY_MAPPING.items():
        if key.lower() == folder_name_lower:
            return value
    
    # Try partial match (e.g., "Ci_001" -> "cirrus")
    for key, value in CATEGORY_MAPPING.items():
        if folder_name_lower.startswith(key.lower()):
            return value
    
    return None

def find_images(directory):
    """
    Recursively find all image files in a directory.
    
    Args:
        directory: Root directory to search
    
    Returns:
        List of (image_path, relative_path) tuples
    """
    image_files = []
    
    if not os.path.exists(directory):
        logger.warning(f"Directory does not exist: {directory}")
        return image_files
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Filter out macOS resource fork files (._*)
            if file.startswith('._'):
                continue
            if Path(file).suffix.lower() in IMAGE_EXTENSIONS:
                full_path = os.path.join(root, file)
                # Get relative path from IMAGE_ROOT_DIR
                relative_path = os.path.relpath(full_path, IMAGE_ROOT_DIR)
                image_files.append((full_path, relative_path))
    
    return image_files

def create_metadata():
    """
    Create metadata.jsonl file with image_id and cloud_category.
    """
    logger.info(f"Scanning CCSN images in: {CCSN_DIR}")
    
    # Find all images
    images = find_images(CCSN_DIR)
    
    if not images:
        logger.warning(f"No images found in {CCSN_DIR}")
        return
    
    logger.info(f"Found {len(images)} images")
    
    # Create metadata entries
    metadata_entries = []
    unknown_categories = set()
    
    for image_path, relative_path in images:
        # Extract folder name (cloud category folder)
        # Path structure: ccsn/Ci/image.jpg -> folder is "Ci"
        path_parts = relative_path.split(os.sep)
        
        if len(path_parts) < 3:
            # Image is directly in ccsn/ folder, no category
            logger.warning(f"Image not in category folder: {relative_path}")
            cloud_category = "unknown"
        else:
            # path_parts[0] = "ccsn", path_parts[1] = category folder
            folder_name = path_parts[1]
            cloud_category = get_cloud_category(folder_name)
            
            if cloud_category is None:
                logger.warning(f"Unknown cloud category for folder '{folder_name}' in {relative_path}")
                unknown_categories.add(folder_name)
                cloud_category = "unknown"
        
        # Create metadata entry
        entry = {
            "image_id": relative_path,
            "cloud_category": cloud_category
        }
        metadata_entries.append(entry)
    
    # Log category statistics
    category_counts = {}
    for entry in metadata_entries:
        category = entry["cloud_category"]
        category_counts[category] = category_counts.get(category, 0) + 1
    
    logger.info("Cloud category distribution:")
    for category, count in sorted(category_counts.items()):
        logger.info(f"  {category}: {count}")
    
    if unknown_categories:
        logger.warning(f"Unknown categories found: {unknown_categories}")
    
    # Write metadata.jsonl
    logger.info(f"Writing metadata to: {METADATA_JSONL}")
    os.makedirs(os.path.dirname(METADATA_JSONL), exist_ok=True)
    
    with open(METADATA_JSONL, 'w') as f:
        for entry in metadata_entries:
            f.write(json.dumps(entry) + '\n')
    
    logger.info(f"Successfully created metadata.jsonl with {len(metadata_entries)} entries")

def main():
    """Main function."""
    logger.info("Starting CCSN metadata generation...")
    create_metadata()
    logger.info("Metadata generation completed!")

if __name__ == "__main__":
    main()

