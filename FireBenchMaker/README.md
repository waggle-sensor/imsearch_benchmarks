# FireBenchMaker

FireBenchMaker is a pipeline for creating **FireBench**, a benchmark dataset for evaluating text-to-image retrieval systems in fire science. The tool uses the [imsearch_benchmaker](https://github.com/waggle-sensor/imsearch_benchmaker) framework to automate the entire dataset creation process, from raw image collection to Hugging Face dataset publication.

## Overview

### What is FireBench?

FireBench is a benchmark dataset for evaluating **text-to-image retrieval systems** in fire science. Given a natural language query like "Fixed long-range daytime webcam images of mountainous shrubland with no visible smoke or flames", the system should retrieve relevant images from a candidate pool.

The dataset is designed to test retrieval systems' ability to:
- Find relevant wildfire-related images based on natural language queries
- Distinguish between smoke and confounders (fog, haze, clouds, glare)
- Handle diverse environmental conditions, viewpoints, and fire stages
- Support fire science research and early detection systems

**Final Dataset**: [FireBench on Hugging Face](https://huggingface.co/datasets/sagecontinuum/FireBench)

## Pipeline Architecture

FireBenchMaker uses the [imsearch_benchmaker](https://github.com/waggle-sensor/imsearch_benchmaker) framework to create the FireBench dataset through a 7-step pipeline.

>NOTE: See [imsearch_benchmaker](https://github.com/waggle-sensor/imsearch_benchmaker) for more details on the pipeline and the tool.

### High-Level Flow

```
Raw Images
    ↓
[1] Preprocess → images.jsonl, seeds.jsonl
    ↓
[2] Vision Annotation → annotations.jsonl (per-image metadata)
    ↓
[3] Query Planning → query_plan.jsonl (candidate pools per query)
    ↓
[4] Judge Generation → firebench_qrels.jsonl (queries + relevance labels)
    ↓
[5] Postprocessing similarity → firebench_qrels_with_clipscore.jsonl (+ CLIP scores)
    ↓
[6] Postprocessing summary → summary/ (visualizations and statistics)
    ↓
[7] Hugging Face upload → hf_dataset/ (ready for upload)
```

### Pipeline Steps

1. **Preprocess**: Converts raw images into JSONL format with metadata (image IDs, licenses, DOIs)
2. **Vision Annotation**: Uses OpenAI's vision API to annotate images with summaries, tags, and categorical facets
3. **Query Planning**: Selects seed images and creates candidate pools (hard/easy/nearmiss negatives) for each query
4. **Judge Generation**: AI fire scientists generate queries and assign binary relevance labels
5. **Postprocessing**: Computes CLIP similarity scores for all query-image pairs and generates exploratory data analysis visualizations and statistics
6. **Hugging Face Upload**: Prepares the dataset in Hugging Face format for publication and uploads to the Hugging Face dataset repository.

## Installation

### Prerequisites

- Python 3.11+
- Access to OpenAI API (for vision annotation and judge generation)
- Kaggle API credentials (for downloading The Wildfire Dataset)
- Sage Continuum credentials (for Sage data collection)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/waggle-sensor/imsearch_benchmarks.git
   cd imsearch_benchmarks/FireBenchMaker
   ```

2. **Install dependencies**:
   ```bash
   # Install main dependencies
   pip install -r requirements.txt
   
   # Install tool dependencies
   pip install -r tools/requirements.txt
   ```

3. **Set up environment variables**:
    Copy the `env.template` file to `env` and fill in the values. Then source the file:
    ```bash
    source env
    ```

## Data Collection

FireBenchMaker includes tools to collect images from three sources:

### 1. HPWREN FIgLib

Downloads fixed long-range webcam imagery from the High-Performance Wireless Research and Education Network.

```bash
python tools/get_figlib.py
```

**Configuration** (in script):
- Sample size: 1300 images
- Output directory: `/tmp/FireBench/images/figlib`
- Random seed: 42

### 2. The Wildfire Dataset

Downloads images from the Kaggle Wildfire Dataset.

```bash
python tools/get_wildfire.py
```

**Configuration** (in script):
- Fire images: 20 samples
- No-fire images: 20 samples
- Output directory: `/tmp/FireBench/images/wildfire`
- Requires Kaggle API credentials

### 3. Sage Continuum

Downloads images from Sage Continuum sensor network.

```bash
python tools/get_sage.py
```

**Configuration** (in script):
- Sample size: 600 images (default)
- Output directory: `/tmp/FireBench/images/sage`
- Requires Sage credentials

>NOTE: The timeframe is set to 2025-12-10 to 2025-12-11 since that was an eruption day for Hawaii's volcano viewable by node W097.

## Configuration

All pipeline configuration is managed through `config.toml`. Key settings include:

### Dataset Metadata
- Benchmark name, description, and author information
- Column name mappings
- Image URL configuration

### Vision Annotation
- Model: `gpt-5-mini` (OpenAI)
- Batch processing settings
- Controlled tag vocabulary (160+ tags)
- Taxonomy definitions (viewpoint, plume_stage, lighting, confounder_type, environment_type)

### Query Planning
- Number of seed images: 100
- Negative sampling strategy:
  - Hard negatives: 25
  - Nearmiss negatives: 10
  - Easy negatives: 5
  - Total negatives: 40 per query

### Judge Generation
- Model: `gpt-5-mini` (OpenAI)
- Binary relevance labeling
- Fire scientist perspective prompts

### Similarity Scoring
- Model: `apple/DFN5B-CLIP-ViT-H-14-378`
- Local CLIP inference

### File Paths
All input/output paths are configurable in `config.toml`. Default paths:
- Images: `/tmp/FireBench/images`
- Inputs: `/Volumes/data/inputs/`
- Outputs: `/Volumes/data/outputs/`

## Running the Pipeline

The pipeline is executed using the `imsearch_benchmaker` CLI. Each step can be run independently:

### Step 1: Preprocess
```bash
benchmaker preprocess
```

Generates:
- `images.jsonl`: Image metadata
- `seeds.jsonl`: Seed images for query generation

### Step 2: Vision Annotation
```bash
benchmaker annotate
```

Generates:
- `annotations.jsonl`: Per-image annotations (summaries, tags, facets, confidence scores)

### Step 3: Query Planning
```bash
benchmaker plan
```

Generates:
- `query_plan.jsonl`: Candidate pools for each query

### Step 4: Judge Generation
```bash
benchmaker judge
```

Generates:
- `firebench_qrels.jsonl`: Queries with relevance labels

### Step 5: Postprocessing
```bash
benchmaker postprocess similarity
benchmaker postprocess summary
```

Generates:
- `firebench_qrels_with_similarity_score.jsonl`: Adds CLIP scores
- `summary/`: Visualizations and statistics

### Step 7: Hugging Face Preparation
```bash
benchmaker upload
```

Generates:
- `hf_dataset/`: Dataset ready for Hugging Face upload

## Output Structure

```
outputs/
├── images.jsonl              # Image metadata
├── seeds.jsonl               # Seed images
├── annotations.jsonl         # Vision annotations
├── query_plan.jsonl          # Query candidate pools
├── firebench_qrels.jsonl     # Queries + relevance labels
├── firebench_qrels_with_similarity_score.jsonl  # With CLIP scores
├── summary/                  # EDA outputs
│   ├── firebench_eda_analysis.ipynb
│   ├── config_values.csv
│   ├── random_image_sample.png
│   └── image_proportion_donuts.png
└── hf_dataset/               # Hugging Face dataset
    └── data/
        └── train-*.parquet
```

## Dataset Details

### Data Fields

Each instance in the final dataset contains:
- `query_id`: Unique query identifier
- `query_text`: Natural language query
- `image_id`: Image identifier
- `relevance_label`: Binary relevance (0 or 1)
- `image`: PIL Image object
- `tags`: Controlled vocabulary tags (12-18 per image)
- `summary`: Brief image description (≤30 words)
- `viewpoint`: Camera viewpoint (fixed_long_range, handheld, aerial, etc.)
- `plume_stage`: Smoke plume stage (incipient, developing, mature, none, etc.)
- `lighting`: Lighting conditions (day, dusk, night, ir_nir, etc.)
- `confounder_type`: Confounder type (cloud, fog, haze, none, etc.)
- `environment_type`: Environment type (forest, grassland, shrubland, etc.)
- `flame_visible`: Boolean indicating flame visibility
- `confidence`: Confidence scores for categorical annotations
- `license`: Image license
- `doi`: Source dataset DOI
- `clip_score`: Pre-computed CLIP similarity score

### Source Datasets

1. **HPWREN FIgLib**: High-Performance Wireless Research and Education Network webcam imagery
2. **The Wildfire Dataset**: Diverse wildfire imagery from Kaggle
3. **Sage Continuum**: Cyberinfrastructure sensor network imagery

See [FireBench on Hugging Face](https://huggingface.co/datasets/sagecontinuum/FireBench) for complete dataset documentation.

## Contact

- **Author**: Francisco Lozano
- **Email**: francisco.lozano@northwestern.edu
- **Affiliation**: Northwestern University
- **GitHub**: [FranciscoLozCoding](https://github.com/FranciscoLozCoding)

## License

The dataset card and annotations are provided under CC BY 4.0. Individual images retain their source licenses.

## Acknowledgments

We thank the creators and maintainers of:
- HPWREN FIgLib
- The Wildfire Dataset