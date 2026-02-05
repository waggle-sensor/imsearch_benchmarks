# CommonObjectsBenchMaker

CommonObjectsBenchMaker is a pipeline for creating **CommonObjectsBench**, a benchmark dataset for evaluating text-to-image retrieval systems on general objects and common scenes. The tool uses the [imsearch_benchmaker](https://github.com/waggle-sensor/imsearch_benchmaker) framework to automate the entire dataset creation process, from raw image collection to Hugging Face dataset publication.

## Overview

### What is CommonObjectsBench?

CommonObjectsBench is a benchmark dataset for evaluating **text-to-image retrieval systems** on general objects and common scenes. Given a natural language query like "A person riding a bicycle on a sunny day", the system should retrieve relevant images from a candidate pool.

The dataset is designed to test retrieval systems' ability to:
- Find relevant images based on natural language queries describing common objects
- Handle diverse scenes, objects, and environments
- Support general-purpose image retrieval research

**Final Public Dataset**: [CommonObjectsBench on Hugging Face](https://huggingface.co/datasets/sagecontinuum/CommonObjectsBench)
**Final Private Dataset**: [CommonObjectsBench-private on Hugging Face](https://huggingface.co/datasets/sagecontinuum/CommonObjectsBench-private)

>NOTE: The private dataset includes imagery from urban sage nodes, which are not allowed in the public dataset. Please be careful when using the private dataset as the urban images are not allowed to be public.

## Directory Structure
```
CommonObjectsBenchMaker/
├── public/
├── private/
├── tools/
└── README.md
└── requirements.txt
└── rights_map.json
```

The `public/` directory contains the configuration and dataset card for the public dataset.
The `private/` directory contains the configuration and dataset card for the private dataset.
The `tools/` directory contains the tools used to collect the images.
The `README.md` file contains the README for the repository.
The `requirements.txt` file contains the requirements for the pipeline.
The `rights_map.json` file contains the rights map for the dataset.

## Pipeline Architecture

CommonObjectsBenchMaker uses the [imsearch_benchmaker](https://github.com/waggle-sensor/imsearch_benchmaker) framework to create the CommonObjectsBench dataset through a 7-step pipeline.

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
[4] Judge Generation → commonobjectsbench_qrels.jsonl (queries + relevance labels)
    ↓
[5] Postprocessing similarity → commonobjectsbench_qrels_with_clipscore.jsonl (+ CLIP scores)
    ↓
[6] Postprocessing summary → summary/ (visualizations and statistics)
    ↓
[7] Hugging Face upload → hf_dataset/ (ready for upload)
```

### Pipeline Steps

1. **Preprocess**: Converts raw images into JSONL format with metadata (image IDs, licenses, DOIs)
2. **Vision Annotation**: Uses OpenAI's vision API to annotate images with summaries, tags, and categorical facets
3. **Query Planning**: Selects seed images and creates candidate pools (hard/easy/nearmiss negatives) for each query
4. **Judge Generation**: AI judges generate queries and assign binary relevance labels
5. **Postprocessing**: Computes CLIP similarity scores for all query-image pairs and generates exploratory data analysis visualizations and statistics
6. **Hugging Face Upload**: Prepares the dataset in Hugging Face format for publication and uploads to the Hugging Face dataset repository.

## Installation

### Prerequisites

- Python 3.11+
- Access to OpenAI API (for vision annotation and judge generation)
- Sage Continuum credentials (for Sage data collection)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/waggle-sensor/imsearch_benchmarks.git
   cd imsearch_benchmarks/CommonObjectsBenchMaker
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

CommonObjectsBenchMaker includes tools to collect images from two sources:

### Dataset Composition

The dataset uses a **70/30 split** between COCO and Sage images:
- **COCO (70%)**: Provides more images and greater diversity
- **Sage (30%)**: Provides real-world imagery with less diversity

> **Note**: This ratio may shift after query generation, as the query planning process selects candidates based on jacard similarity.

### 1. COCO 2017 Dataset

Downloads images from the COCO (Common Objects in Context) 2017 dataset using FiftyOne.

```bash
python tools/get_coco.py
```

**Configuration** (in script):
- COCO classes: Configurable list (default: all 80 COCO classes)
- Sample size per class: Configurable
- Output directory: `/tmp/CommonObjectsBench/images/coco`
- Random seed: 42

### 2. Sage Continuum

Downloads random images from Sage Continuum sensor network.

```bash
python tools/get_sage.py
```

**Configuration** (in script):
- **Time frame**: 
  - Start date: 2 years after the "start of Sage" date
  - End date: 1 week before the current date (to avoid processing delays)
- **Time slot configuration**: 
  - Calculated dynamically based on total days between start and end dates
  - Approximately 1 time slot per week
  - Minimum: 20 slots for short ranges
  - Maximum: Capped to avoid excessive queries
  - Randomly distributed across the time range for temporal diversity
- **VSN list**: 
  - Default: `None` (queries all available VSNs)
  - Urban nodes are not included by default (see dataset privacy notes below). To include urban nodes, set the `SAGE_URBAN_IMAGERY` environment variable to `true`.
- **Sample size**: Configurable (default: 1000)
- **Output directory**: `/tmp/CommonObjectsBenchMaker/images/sage`
- **Image filtering**: 
  - Top camera images are automatically excluded (they typically show only sky with no objects)
- **Requires Sage credentials**

## Configuration

All pipeline configuration is managed through `config.toml`. Key settings include:

### Dataset Metadata
- Benchmark name, description, and author information
- Column name mappings
- Image URL configuration

### Vision Annotation
- Model: `gpt-5-mini` (OpenAI)
- Batch processing settings
- Controlled tag vocabulary (general objects and scenes)

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

### Similarity Scoring
- Model: `apple/DFN5B-CLIP-ViT-H-14-378`
- Local CLIP inference

### File Paths
All input/output paths are configurable in `config.toml`. Default paths:
- Images: `/tmp/CommonObjectsBenchMaker/images`
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
- `commonobjectsbench_qrels.jsonl`: Queries with relevance labels

### Step 5: Postprocessing
```bash
benchmaker postprocess similarity
benchmaker postprocess summary
```

Generates:
- `commonobjectsbench_qrels_with_similarity_score.jsonl`: Adds CLIP scores
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
├── commonobjectsbench_qrels.jsonl     # Queries + relevance labels
├── commonobjectsbench_qrels_with_similarity_score.jsonl  # With CLIP scores
├── summary/                  # EDA outputs
│   ├── commonobjectsbench_eda_analysis.ipynb
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
- `confidence`: Confidence scores for annotations
- `license`: Image license
- `doi`: Source dataset DOI
- `clip_score`: Pre-computed CLIP similarity score

### Source Datasets

1. **COCO Dataset**: Common Objects in Context dataset
2. **Sage Continuum**: Cyberinfrastructure sensor network imagery

## Contact

- **Author**: Francisco Lozano
- **Email**: francisco.lozano@northwestern.edu
- **Affiliation**: Northwestern University
- **GitHub**: [FranciscoLozCoding](https://github.com/FranciscoLozCoding)

## License

The dataset card and annotations are provided under CC BY 4.0. Individual images retain their source licenses.

## Acknowledgments

We thank the creators and maintainers of:
- COCO Dataset
- Sage Continuum

## Dataset Privacy and Versions

Two versions of the dataset are created:

1. **CommonObjectsBench-private**: Includes imagery from urban Sage nodes (not allowed in public datasets)
2. **CommonObjectsBench**: Public version that excludes urban Sage nodes

> **Important**: The private dataset includes urban imagery that cannot be made public. Use with caution and ensure compliance with Sage Continuum data usage policies.

## Dataset Creation Notes

### Known Issues and Data Loss

During dataset creation, some data loss occurred due to various technical issues:

#### CommonObjectsBench-private
- **Vision Annotation**: Lost 190 images due to max tokens error in the vision model
  - The model continued returning `\n` or `\r` characters after completing its expected output
- **Query Planning**: Lost 9 queries because their seed images failed in the vision annotation step
- **Similarity Postprocessing**: Lost ~22 query/image pairs due to filesystem issues on the image hosting machine

#### CommonObjectsBench (Public)
- **Vision Annotation**: Lost 152 images due to max tokens error in the vision model
  - Same issue as private dataset: model returned `\n` or `\r` characters after expected output
- **Query Planning**: Lost 3 queries because their seed images failed in the vision annotation step
- **Similarity Postprocessing**: Lost 18 query/image pairs due to filesystem issues on the image hosting machine

### Dataset Characteristics

- **Relevance Distribution**: The relevancy distribution is skewed towards non-relevant images
  - This is expected because COCO is highly diverse with many images that don't relate to each other