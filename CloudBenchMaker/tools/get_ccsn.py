#!/usr/bin/env python3
"""
get_ccsn.py

Download all images from CCSN dataset for the CloudBench benchmark and create metadata.
The CCSN (Cirrus Cumulus Stratus Nimbus) dataset is hosted at Harvard Dataverse.
"""
import requests
import os
import zipfile
import shutil
from pathlib import Path
import logging
from PIL import Image
import random
from ccsn_meta import create_metadata

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DATAVERSE_BASE_URL = "https://dataverse.harvard.edu"
DATASET_PERSISTENT_ID = "doi:10.7910/DVN/CADDPD"
DATAVERSE_API_TOKEN = os.getenv("DATAVERSE_API_TOKEN", "")  # Optional API token
OUTPUT_DIR = "/tmp/CloudBench/images/ccsn"
TEMP_DIR = "/tmp/CloudBench/temp/ccsn"
RANDOM_SEED = 14
SAMPLE_SIZE = 2543  # Total number of images to sample (50% of CloudBench)

def get_dataset_files(persistent_id, api_token=""):
    """
    Get list of files in the dataset using Dataverse API.
    
    Args:
        persistent_id: Dataset persistent ID (DOI)
        api_token: Optional API token for authentication
    
    Returns:
        List of file metadata dictionaries
    """
    url = f"{DATAVERSE_BASE_URL}/api/datasets/:persistentId"
    params = {"persistentId": persistent_id}
    headers = {}
    
    if api_token:
        headers["X-Dataverse-key"] = api_token
    
    try:
        logger.info(f"Fetching dataset metadata from Dataverse...")
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        dataset_info = response.json()
        
        # Get the dataset ID
        dataset_id = dataset_info["data"]["id"]
        logger.info(f"Dataset ID: {dataset_id}")
        
        # Get files in the dataset
        files_url = f"{DATAVERSE_BASE_URL}/api/datasets/{dataset_id}/versions/:latest/files"
        files_response = requests.get(files_url, headers=headers, timeout=30)
        files_response.raise_for_status()
        files_data = files_response.json()
        
        return files_data["data"]
    except Exception as e:
        logger.error(f"Failed to fetch dataset files: {e}")
        raise

def download_file(file_id, filename, output_path, api_token=""):
    """
    Download a file from Dataverse by file ID.
    
    Args:
        file_id: Dataverse file ID
        filename: Original filename
        output_path: Path to save the file
        api_token: Optional API token
    
    Returns:
        True if successful, False otherwise
    """
    url = f"{DATAVERSE_BASE_URL}/api/access/datafile/{file_id}"
    headers = {}
    
    if api_token:
        headers["X-Dataverse-key"] = api_token
    
    try:
        logger.info(f"Downloading {filename}...")
        response = requests.get(url, headers=headers, stream=True, timeout=300)
        response.raise_for_status()
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Download with progress
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        if downloaded % (1024 * 1024) == 0:  # Log every MB
                            logger.info(f"Downloaded {downloaded / (1024 * 1024):.1f} MB / {total_size / (1024 * 1024):.1f} MB ({percent:.1f}%)")
        
        logger.info(f"Successfully downloaded {filename} ({downloaded / (1024 * 1024):.1f} MB)")
        return True
    except Exception as e:
        logger.error(f"Failed to download {filename}: {e}")
        return False

def extract_zip(zip_path, extract_dir):
    """
    Extract ZIP file to directory.
    
    Args:
        zip_path: Path to ZIP file
        extract_dir: Directory to extract to
    
    Returns:
        Path to extracted directory
    """
    try:
        logger.info(f"Extracting {zip_path} to {extract_dir}...")
        os.makedirs(extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        logger.info(f"Successfully extracted to {extract_dir}")
        return extract_dir
    except Exception as e:
        logger.error(f"Failed to extract ZIP file: {e}")
        raise

def find_image_files(directory):
    """
    Recursively find all image files in a directory.
    
    Args:
        directory: Root directory to search
    
    Returns:
        List of image file paths
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff', '.webp'}
    image_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    
    return image_files

def verify_image(image_path):
    """
    Verify that a file is a valid image.
    
    Args:
        image_path: Path to image file
    
    Returns:
        True if valid image, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def copy_images(image_files, output_dir, sample_size=None, random_seed=RANDOM_SEED):
    """
    Copy image files to output directory, optionally sampling.
    
    Args:
        image_files: List of image file paths
        output_dir: Output directory
        sample_size: Optional number of images to sample
        random_seed: Random seed for sampling
    
    Returns:
        Dictionary with copy results
    """
    # Filter valid images
    logger.info(f"Verifying {len(image_files)} image files...")
    valid_images = []
    for img_path in image_files:
        if verify_image(img_path):
            valid_images.append(img_path)
        else:
            logger.warning(f"Invalid image: {img_path}")
    
    logger.info(f"Found {len(valid_images)} valid images")
    
    # Sample if needed
    if sample_size and len(valid_images) > sample_size:
        logger.info(f"Sampling {sample_size} images from {len(valid_images)} total")
        random.seed(random_seed)
        sampled_images = random.sample(valid_images, sample_size)
    else:
        logger.info(f"Using all {len(valid_images)} images")
        sampled_images = valid_images
    
    # Copy images
    os.makedirs(output_dir, exist_ok=True)
    results = {"success": 0, "error": 0}
    
    for img_path in sampled_images:
        try:
            # Get relative path from the extracted directory to preserve structure
            # If images are in subdirectories (by category), preserve that structure
            filename = os.path.basename(img_path)
            
            # Check if image is in a subdirectory (category folder)
            parent_dir = os.path.basename(os.path.dirname(img_path))
            
            # Create output path
            if parent_dir and parent_dir not in ['ccsn', 'CCSN']:  # Preserve category structure
                output_subdir = os.path.join(output_dir, parent_dir)
                os.makedirs(output_subdir, exist_ok=True)
                output_path = os.path.join(output_subdir, filename)
            else:
                output_path = os.path.join(output_dir, filename)
            
            # Skip if already exists
            if os.path.exists(output_path):
                logger.debug(f"Skipping existing file: {output_path}")
                results["success"] += 1
                continue
            
            # Copy file
            shutil.copy2(img_path, output_path)
            results["success"] += 1
            
            if results["success"] % 100 == 0:
                logger.info(f"Copied {results['success']} images...")
                
        except Exception as e:
            logger.error(f"Failed to copy {img_path}: {e}")
            results["error"] += 1
    
    return results

def main():
    """Main function to download CCSN images."""
    logger.info("Starting CCSN image download...")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Sample size: {SAMPLE_SIZE}")
    
    # Create temp directory
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    try:
        # Get dataset files
        files = get_dataset_files(DATASET_PERSISTENT_ID, DATAVERSE_API_TOKEN)
        
        if not files:
            logger.error("No files found in dataset")
            return
        
        # Find the ZIP file (usually named CCSN.zip or similar)
        zip_file = None
        for file_info in files:
            filename = file_info["dataFile"]["filename"]
            if filename.lower().endswith('.zip'):
                zip_file = file_info
                logger.info(f"Found ZIP file: {filename}")
                break
        
        if not zip_file:
            logger.error("No ZIP file found in dataset")
            return
        
        # Download ZIP file
        zip_filename = zip_file["dataFile"]["filename"]
        zip_file_id = zip_file["dataFile"]["id"]
        zip_path = os.path.join(TEMP_DIR, zip_filename)
        
        if not download_file(zip_file_id, zip_filename, zip_path, DATAVERSE_API_TOKEN):
            logger.error("Failed to download ZIP file")
            return
        
        # Extract ZIP file
        extract_dir = os.path.join(TEMP_DIR, "extracted")
        extract_zip(zip_path, extract_dir)
        
        # Find all image files
        image_files = find_image_files(extract_dir)
        logger.info(f"Found {len(image_files)} image files in ZIP")
        
        if not image_files:
            logger.error("No image files found in extracted ZIP")
            return
        
        # Copy images to output directory
        results = copy_images(image_files, OUTPUT_DIR, sample_size=SAMPLE_SIZE, random_seed=RANDOM_SEED)
        
        logger.info(f"Download completed: {results['success']} success, {results['error']} errors")
        logger.info(f"Images saved to: {OUTPUT_DIR}")

        # Create metadata
        logger.info("Starting CCSN metadata generation...")
        create_metadata()
        logger.info("Metadata generation completed!")
        
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        raise
    finally:
        # Clean up temp directory
        if os.path.exists(TEMP_DIR):
            logger.info(f"Cleaning up temp directory: {TEMP_DIR}")
            shutil.rmtree(TEMP_DIR)

if __name__ == "__main__":
    main()
