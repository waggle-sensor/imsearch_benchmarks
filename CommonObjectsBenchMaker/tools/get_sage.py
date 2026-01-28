#!/usr/bin/env python3
"""
get_sage.py

Download random images from Sage Continuum sensor network for the CommonObjectsBench benchmark.
Supports configurable time frames, VSN lists, and random time slot sampling.
"""
import sage_data_client
import requests
from PIL import Image
from io import BytesIO
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from urllib.parse import urlparse
import pandas as pd
import random
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Auth header for Sage
USER = os.getenv("SAGE_USER", "")
PASSWORD = os.getenv("SAGE_PASSWORD", "")
auth = (USER, PASSWORD)

# Number of concurrent downloads
MAX_WORKERS = 10

# Configuration
# Time frame for querying (ISO format: "YYYY-MM-DDTHH:MM:SS.000Z")
# Set these to your desired time range
#TODO: Set these to start of sage to now
TIME_FRAME_START = "2024-01-01T00:00:00.000Z"  # Start date
TIME_FRAME_END = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.000Z")    # End date

# Time slot configuration
#TODO: Configure these
NUM_TIME_SLOTS = 10  # Number of random time slots to sample
TIME_SLOT_DURATION_HOURS = 1  # Duration of each time slot in hours

# VSN configuration
# Set to None to query all available VSNs, or provide a list like ["W097", "W019", "W020"]
#TODO: Set these to the VSNs you want to query
VSN_LIST = None  # None = query all available VSNs

# Image task types to query
#TODO: Set these to all the image tasks
IMAGE_TASKS = ["imagesampler-mobotix", "imagesampler-bottom"]

# Sample size
#TODO: Set these to the number of images you want to use
SAMPLE_SIZE = 1000  # Total number of images to sample

# Output configuration
OUTPUT_DIR = "/tmp/CommonObjectsBenchMaker/images/sage"
RANDOM_SEED = 42

def download_image(session, image_url, filepath, auth):
    """Download and save a single image."""
    try:
        # Skip if file already exists
        if os.path.exists(filepath):
            return {"status": "skipped", "filepath": filepath}
        
        # Ensure directory exists (if filepath has a directory component)
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        
        # Download image
        response = session.get(image_url, auth=auth, timeout=30)
        response.raise_for_status()
        
        # Save image
        image = Image.open(BytesIO(response.content))
        image.save(filepath)
        
        return {"status": "success", "filepath": filepath}
    except Exception as e:
        logger.error(f"Failed to download {image_url}: {e}")
        return {"status": "error", "filepath": filepath, "error": str(e)}

def extract_path_from_url(url, base_url="https://storage.sagecontinuum.org/api/v1/data"):
    """Extract the path from URL excluding the base URL."""
    if url.startswith(base_url):
        # Remove the base URL and get the path
        path = url[len(base_url):].lstrip('/')
        return path
    else:
        # If URL doesn't match expected format, parse it
        parsed = urlparse(url)
        path = parsed.path.lstrip('/')
        # Remove /api/v1/data prefix if present
        if path.startswith('api/v1/data/'):
            path = path[len('api/v1/data/'):]
        return path

def download_images(df, output_dir, auth, max_workers=MAX_WORKERS):
    """Download images from a dataframe in parallel."""
    # Create session for connection pooling
    session = requests.Session()
    
    # Prepare download tasks
    tasks = []
    for _, row in df.iterrows():
        image_url = row.value
        # Extract path from URL (excluding base URL)
        relative_path = extract_path_from_url(image_url)
        # Combine with output directory
        filepath = os.path.join(output_dir, relative_path)
        # Create directory structure if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        tasks.append((image_url, filepath))
    
    logger.info(f"Starting download of {len(tasks)} images to {output_dir}")
    
    # Download in parallel
    results = {"success": 0, "skipped": 0, "error": 0}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [
            executor.submit(download_image, session, url, path, auth)
            for url, path in tasks
        ]
        
        # Process completed downloads
        for future in as_completed(futures):
            result = future.result()
            results[result["status"]] += 1
    
    session.close()
    logger.info(f"Completed {output_dir}: {results['success']} success, "
                f"{results['skipped']} skipped, {results['error']} errors")
    return results

def generate_random_time_slots(start_str, end_str, num_slots, duration_hours, random_seed=RANDOM_SEED):
    """Generate random time slots within a time frame."""
    random.seed(random_seed)
    
    # Parse time strings
    start = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
    end = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
    
    duration = timedelta(hours=duration_hours)
    total_duration = end - start
    
    if total_duration < duration:
        logger.warning(f"Time frame ({total_duration}) is shorter than slot duration ({duration}). Using single slot.")
        return [(start_str, end_str)]
    
    # Generate random time slots
    slots = []
    for _ in range(num_slots):
        # Random start time within the time frame
        max_start = end - duration
        if max_start <= start:
            slot_start = start
            slot_end = end
        else:
            # Random seconds offset
            random_seconds = random.randint(0, int((max_start - start).total_seconds()))
            slot_start = start + timedelta(seconds=random_seconds)
            slot_end = slot_start + duration
            
            # Ensure slot doesn't exceed end time
            if slot_end > end:
                slot_end = end
                slot_start = slot_end - duration
        
        # Convert back to ISO format
        slot_start_str = slot_start.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        slot_end_str = slot_end.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        slots.append((slot_start_str, slot_end_str))
    
    return slots

def query_sage_images(time_slots, vsn_list, image_tasks):
    """Query Sage images for given time slots and VSNs."""
    all_dfs = []
    
    for slot_start, slot_end in time_slots:
        logger.info(f"Querying time slot: {slot_start} to {slot_end}")
        
        for task in image_tasks:
            if vsn_list is None:
                # Query all VSNs (no vsn filter)
                try:
                    df = sage_data_client.query(
                        start=slot_start,
                        end=slot_end,
                        filter={"task": task}
                    )
                    if df is not None and len(df) > 0:
                        logger.info(f"Found {len(df)} images for task '{task}' in time slot")
                        all_dfs.append(df)
                except Exception as e:
                    logger.warning(f"Failed to query task '{task}' for time slot: {e}")
            else:
                # Query specific VSNs
                for vsn in vsn_list:
                    try:
                        df = sage_data_client.query(
                            start=slot_start,
                            end=slot_end,
                            filter={
                                "vsn": vsn,
                                "task": task
                            }
                        )
                        if df is not None and len(df) > 0:
                            logger.info(f"Found {len(df)} images for VSN '{vsn}', task '{task}' in time slot")
                            all_dfs.append(df)
                    except Exception as e:
                        logger.warning(f"Failed to query VSN '{vsn}', task '{task}' for time slot: {e}")
    
    if len(all_dfs) == 0:
        logger.warning("No images found in any time slots")
        return pd.DataFrame()
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Remove duplicates if any
    if 'value' in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset=['value'])
    
    logger.info(f"Total unique images found: {len(combined_df)}")
    return combined_df

def main():
    """Main function to download Sage images."""
    random.seed(RANDOM_SEED)
    logger.info(f"Using random seed: {RANDOM_SEED}")
    
    # Determine VSN list
    if VSN_LIST is None:
        logger.info("VSN_LIST is None, querying all VSNs...")
        vsn_list = None
    else:
        vsn_list = VSN_LIST
        logger.info(f"Using specified VSN list: {vsn_list}")
    
    # Generate random time slots
    logger.info(f"Generating {NUM_TIME_SLOTS} random time slots...")
    logger.info(f"Time frame: {TIME_FRAME_START} to {TIME_FRAME_END}")
    time_slots = generate_random_time_slots(
        TIME_FRAME_START,
        TIME_FRAME_END,
        NUM_TIME_SLOTS,
        TIME_SLOT_DURATION_HOURS,
        random_seed=RANDOM_SEED
    )
    logger.info(f"Generated {len(time_slots)} time slots")
    
    # Query Sage images
    logger.info("Querying Sage data...")
    df = query_sage_images(time_slots, vsn_list, IMAGE_TASKS)
    
    if len(df) == 0:
        logger.error("No images found. Please check your time frame and VSN configuration.")
        return
    
    # Sample images if needed
    if len(df) > SAMPLE_SIZE:
        logger.info(f"Sampling {SAMPLE_SIZE} images from {len(df)} total")
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
    else:
        logger.info(f"Using all {len(df)} images (less than sample size {SAMPLE_SIZE})")
    
    # Download images in parallel
    logger.info(f"Downloading {len(df)} images...")
    download_images(df, OUTPUT_DIR, auth)
    
    logger.info("All downloads completed!")

if __name__ == "__main__":
    main()

