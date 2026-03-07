#!/usr/bin/env python3
"""
get_sage.py

Download random images from Sage Continuum sensor network for the SageBench benchmark.
Supports configurable time frames, VSN lists, and random time slot sampling.
Writes metadata.jsonl with Sage metadata (vsn, zone, host, job, plugin, camera, project, address)
so the benchmark can evaluate retrieval when queries reference this metadata.
"""
import json
import sage_data_client
import requests
from PIL import Image
from io import BytesIO
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from urllib.parse import urlparse, urljoin
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

# Sage metadata columns to persist (must match vision_metadata_columns in config.toml)
SAGE_METADATA_KEYS = ["vsn", "zone", "host", "job", "plugin", "camera", "project", "address"]

# Configuration
# Time frame for querying (ISO format: "YYYY-MM-DDTHH:MM:SS.000Z")
TIME_FRAME_START = "2024-01-01T00:00:00.000Z"
TIME_FRAME_END = (datetime.now() - timedelta(weeks=1)).strftime("%Y-%m-%dT%H:%M:%S.000Z")

TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.000Z"
time_start = datetime.strptime(TIME_FRAME_START, TIME_FORMAT)
time_end = datetime.strptime(TIME_FRAME_END, TIME_FORMAT)
total_days = (time_end - time_start).days

NUM_TIME_SLOTS = 200
TIME_SLOT_DURATION_HOURS = 0.5

# Manifest API (project and address are not in query meta; fetch from manifest per VSN)
MANIFEST_API = os.environ.get("MANIFEST_API", "https://auth.sagecontinuum.org/manifests/")

# VSN configuration
SAGE_URBAN_IMAGERY = os.getenv("SAGE_URBAN_IMAGERY", "false").lower()
if SAGE_URBAN_IMAGERY == "true":
    UNALLOWED_NODES = []
else:
    UNALLOWED_NODES = [
        "N001", "V012", "V027", "V028", "V042", "W015", "W019", "W01C", "W01E",
        "W022", "W023", "W024", "W026", "W027", "W028", "W029", "W02B", "W02C",
        "W02D", "W02E", "W02F", "W031", "W040", "W042", "W044", "W045", "W046",
        "W047", "W048", "W049", "W04A", "W04B", "W04C", "W04D", "W04E", "W04F",
        "W050", "W051", "W052", "W053", "W054", "W055", "W056", "W059", "W05A",
        "W05B", "W05C", "W05D", "W05E", "W05F", "W060", "W061", "W062", "W063",
        "W064", "W065", "W066", "W068", "W06B", "W06D", "W06E", "W071", "W072",
        "W073", "W074", "W075", "W076", "W077", "W078", "W079", "W07A", "W07B",
        "W07C", "W07D", "W07E", "W07F", "W080", "W081", "W082", "W085", "W086",
        "W087", "W088", "W089", "W08A", "W08B", "W08D", "W08E", "W08F", "W090",
        "W091", "W092", "W093", "W094", "W095", "W096", "W098", "W099", "W09A",
        "W09B", "W09C", "W09D", "W09E", "W09F", "W0A0", "W0A1", "W0A2", "W0A3",
        "W0A4", "W0A5", "W0AB", "W0AD", "W0BB", "W0BC", "W0BD", "W0BF", "W0AA"
    ]
UNALLOWED_NODES_SET = set(vsn.lower() for vsn in UNALLOWED_NODES)

IMAGE_TASKS = "imagesampler-.*"
SAMPLE_SIZE = 5000

# Output configuration: images under image_root/sage/, metadata at image_root/metadata.jsonl
OUTPUT_DIR = "/tmp/SageBench/images/sage"
IMAGE_ROOT_DIR = os.path.dirname(OUTPUT_DIR)
METADATA_JSONL = os.path.join(IMAGE_ROOT_DIR, "metadata.jsonl")
RANDOM_SEED = 23


def download_image(session, image_url, filepath, auth):
    """Download and save a single image."""
    try:
        if os.path.exists(filepath):
            return {"status": "skipped", "filepath": filepath}
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        response = session.get(image_url, auth=auth, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        image.save(filepath)
        return {"status": "success", "filepath": filepath}
    except Exception as e:
        logger.error(f"Failed to download {image_url}: {e}")
        return {"status": "error", "filepath": filepath, "error": str(e)}


def extract_path_from_url(url, base_url="https://storage.sagecontinuum.org/api/v1/data"):
    """Extract the path from URL excluding the base URL."""
    if url.startswith(base_url):
        path = url[len(base_url):].lstrip('/')
        return path
    parsed = urlparse(url)
    path = parsed.path.lstrip('/')
    if path.startswith('api/v1/data/'):
        path = path[len('api/v1/data/'):]
    return path


def download_images(df, output_dir, auth, max_workers=MAX_WORKERS):
    """Download images from a dataframe in parallel."""
    session = requests.Session()
    tasks = []
    for _, row in df.iterrows():
        image_url = row.value
        relative_path = extract_path_from_url(image_url)
        filepath = os.path.join(output_dir, relative_path)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        tasks.append((image_url, filepath))
    logger.info(f"Starting download of {len(tasks)} images to {output_dir}")
    results = {"success": 0, "skipped": 0, "error": 0}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(download_image, session, url, path, auth)
            for url, path in tasks
        ]
        for future in as_completed(futures):
            result = future.result()
            results[result["status"]] += 1
    session.close()
    logger.info(f"Completed {output_dir}: {results['success']} success, "
                f"{results['skipped']} skipped, {results['error']} errors")
    return results


def get_manifest_for_vsn(vsn):
    """Fetch manifest for a VSN and return (project, address). project/address are not in query meta."""
    if not vsn or (isinstance(vsn, float) and pd.isna(vsn)):
        return "unknown", "unknown"
    vsn = str(vsn).strip().upper()
    try:
        response = requests.get(urljoin(MANIFEST_API, vsn), timeout=10)
        response.raise_for_status()
        manifest = response.json()
        project = manifest.get("project", "unknown")
        address = manifest.get("address", "unknown")
        return (
            "" if project is None else str(project).strip(),
            "" if address is None else str(address).strip(),
        )
    except Exception as e:
        logger.warning(f"Could not fetch manifest for VSN {vsn}: {e}")
        return "unknown", "unknown"


def fetch_manifest_cache(df):
    """Build vsn -> (project, address) from manifest API for all unique VSNs in df."""
    vsn_col = "meta.vsn" if "meta.vsn" in df.columns else None
    if vsn_col is None:
        return {}
    unique_vsns = df[vsn_col].dropna().astype(str).str.strip().str.upper().unique().tolist()
    cache = {}
    for vsn in unique_vsns:
        if vsn and vsn != "NAN":
            cache[vsn] = get_manifest_for_vsn(vsn)
    logger.info(f"Fetched manifest (project, address) for {len(cache)} unique VSNs")
    return cache


def write_metadata_jsonl(df, output_dir, metadata_path):
    """
    Write metadata.jsonl with image_id (relative to image root) and Sage metadata columns.
    image_id uses 'sage/' prefix so paths are relative to image_root_dir.
    project and address are fetched from the manifest API (not in query meta).
    """
    manifest_cache = fetch_manifest_cache(df)
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w") as f:
        for _, row in df.iterrows():
            relative_path = extract_path_from_url(row["value"])
            image_id = os.path.join("sage", relative_path).replace("\\", "/")
            entry = {"image_id": image_id}
            for key in SAGE_METADATA_KEYS:
                if key == "project":
                    vsn = row["meta.vsn"] if "meta.vsn" in row.index else None
                    vsn_key = str(vsn).strip().upper() if vsn is not None and not pd.isna(vsn) else ""
                    proj, addr = manifest_cache.get(vsn_key, ("unknown", "unknown"))
                    entry["project"] = proj
                    entry["address"] = addr
                    continue
                if key == "address":
                    continue  # already set with project
                meta_col = f"meta.{key}"
                if meta_col in row.index:
                    val = row[meta_col]
                    entry[key] = "" if pd.isna(val) else str(val).strip()
                else:
                    entry[key] = ""
            f.write(json.dumps(entry) + "\n")
    logger.info(f"Wrote metadata for {len(df)} images to {metadata_path}")


def generate_random_time_slots(start_str, end_str, num_slots, duration_hours, random_seed=RANDOM_SEED):
    """Generate random time slots within a time frame."""
    random.seed(random_seed)
    start = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
    end = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
    duration = timedelta(hours=duration_hours)
    total_duration = end - start
    if total_duration < duration:
        return [(start_str, end_str)]
    slots = []
    for _ in range(num_slots):
        max_start = end - duration
        if max_start <= start:
            slot_start, slot_end = start, end
        else:
            random_seconds = random.randint(0, int((max_start - start).total_seconds()))
            slot_start = start + timedelta(seconds=random_seconds)
            slot_end = slot_start + duration
            if slot_end > end:
                slot_end = end
                slot_start = slot_end - duration
        slots.append((
            slot_start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            slot_end.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        ))
    return slots


def check_url_accessible(url, auth, timeout=10):
    """Check if a URL is accessible and returns a valid image file."""
    try:
        response = requests.head(url, auth=auth, timeout=timeout, allow_redirects=True)
        if response.status_code not in [200, 302]:
            return False
        response = requests.get(url, auth=auth, timeout=timeout, stream=True)
        if response.status_code != 200:
            return False
        chunk_size = 32768
        content_chunk = b''
        for chunk in response.iter_content(chunk_size=chunk_size):
            content_chunk += chunk
            if len(content_chunk) >= chunk_size:
                break
        try:
            image = Image.open(BytesIO(content_chunk))
            image.verify()
            return True
        except Exception:
            return False
    except Exception:
        return False


def filter_accessible_urls(df, auth):
    """Filter dataframe to keep only rows with accessible URLs that are valid images."""
    if 'value' not in df.columns:
        logger.warning("No 'value' column found in dataframe, skipping URL validation")
        return df
    original_count = len(df)
    logger.info(f"Validating URLs and image format for {original_count} images...")
    valid_indices = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(check_url_accessible, row['value'], auth): idx
                  for idx, row in df.iterrows()}
        for future in as_completed(futures):
            idx = futures[future]
            if future.result():
                valid_indices.append(idx)
    df_filtered = df.loc[valid_indices]
    logger.info(f"Found {len(df_filtered)} images with accessible and valid image URLs "
                f"(removed {original_count - len(df_filtered)} invalid/inaccessible)")
    return df_filtered


def query_sage_images(time_slots: list[tuple[str, str]]) -> pd.DataFrame:
    """Query Sage images for given time slots."""
    all_dfs = []
    for slot_start, slot_end in time_slots:
        logger.info(f"Querying time slot: {slot_start} to {slot_end}")
        try:
            df = sage_data_client.query(
                start=slot_start,
                end=slot_end,
                filter={"task": IMAGE_TASKS}
            )
            if len(df) <= 0:
                logger.warning(f"No images found for time slot: {slot_start} to {slot_end}")
                continue
            df = df[df['meta.vsn'].apply(lambda x: x.strip().lower() not in UNALLOWED_NODES_SET)]
            if len(df) <= 0:
                logger.warning(f"No images found for time slot after removing urban nodes")
                continue
            pil_valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff', '.webp')
            df = df[df['value'].str.lower().str.endswith(pil_valid_extensions)]
            if len(df) <= 0:
                continue
            df = filter_accessible_urls(df, auth)
            if len(df) <= 0:
                continue
            all_dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to query time slot: {slot_start} to {slot_end}: {e}")
            continue
    if len(all_dfs) == 0:
        logger.warning("No images found in any time slots")
        return pd.DataFrame()
    combined_df = pd.concat(all_dfs, ignore_index=True)
    if 'value' in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset=['value'])
    logger.info(f"Total unique images found: {len(combined_df)}")
    return combined_df


def main():
    """Main function to download Sage images and write metadata.jsonl."""
    random.seed(RANDOM_SEED)
    logger.info(f"Using random seed: {RANDOM_SEED}")
    if SAGE_URBAN_IMAGERY == "true":
        logger.info("Urban imagery is enabled, urban VSNs will be queried...")
    else:
        logger.info("Urban imagery is disabled, no urban VSNs will be queried...")
    logger.info(f"Generating {NUM_TIME_SLOTS} random time slots...")
    time_slots = generate_random_time_slots(
        TIME_FRAME_START, TIME_FRAME_END,
        NUM_TIME_SLOTS, TIME_SLOT_DURATION_HOURS,
        random_seed=RANDOM_SEED
    )
    logger.info("Querying Sage data...")
    df = query_sage_images(time_slots)
    if len(df) == 0:
        logger.error("No images found. Please check your time frame and VSN configuration.")
        return
    if len(df) > SAMPLE_SIZE:
        logger.info(f"Sampling {SAMPLE_SIZE} images from {len(df)} total")
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
    else:
        logger.info(f"Using all {len(df)} images (less than sample size {SAMPLE_SIZE})")
    logger.info(f"Downloading {len(df)} images...")
    download_images(df, OUTPUT_DIR, auth)
    logger.info(f"Writing metadata to {METADATA_JSONL}")
    write_metadata_jsonl(df, OUTPUT_DIR, METADATA_JSONL)
    logger.info("All downloads and metadata export completed!")


if __name__ == "__main__":
    main()
