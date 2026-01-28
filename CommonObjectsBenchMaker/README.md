# CommonObjectsBenchMaker

CommonObjectsBenchMaker is a pipeline for creating **CommonObjectsBench**, a benchmark dataset for evaluating text-to-image retrieval systems on general objects and common scenes. The tool uses the [imsearch_benchmaker](https://github.com/waggle-sensor/imsearch_benchmaker) framework to automate the entire dataset creation process, from raw image collection to Hugging Face dataset publication.

## Overview

### What is CommonObjectsBench?

CommonObjectsBench is a benchmark dataset for evaluating **text-to-image retrieval systems** on general objects and common scenes. Given a natural language query like "A person riding a bicycle on a sunny day", the system should retrieve relevant images from a candidate pool.

The dataset is designed to test retrieval systems' ability to:
- Find relevant images based on natural language queries describing common objects
- Handle diverse scenes, objects, and environments
- Support general-purpose image retrieval research

**Final Dataset**: [CommonObjectsBench on Hugging Face](https://huggingface.co/datasets/sagecontinuum/CommonObjectsBench)

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

### 1. COCO Dataset

Downloads images from the COCO (Common Objects in Context) dataset using FiftyOne.

```bash
python tools/get_coco.py
```

**Configuration** (in script):
- COCO classes: Configurable list (default: all 80 COCO classes)
- Sample size per class: Configurable
- Output directory: `/tmp/CommonObjectsBenchMaker/images/coco`
- Random seed: 42

### 2. Sage Continuum

Downloads random images from Sage Continuum sensor network.

```bash
python tools/get_sage.py
```

**Configuration** (in script):
- Time frame: Configurable start and end dates
- Number of time slots: Configurable (default: 10)
- VSN list: Configurable list of VSNs (default: query all available)
- Sample size: Configurable (default: 1000)
- Output directory: `/tmp/CommonObjectsBenchMaker/images/sage`
- Requires Sage credentials

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



# NOTES TO SELF
>NOTE: later organize these note into the README.md file

- the split of coco and sage is a 70/30 split initially, with majority of the images coming from coco
    - the reason for this is because coco has more images and is more diverse, but sage has more images from the real world but less diverse
    - Although this can shift once the queries are generated
- the time frame start date in get_sage.py was calcualted by adding 2 years to the "start of sage" date
- the time frame end date in get_sage.py was calculated by subtracting 1 week from the current date
    - 1 week is to avoid running into errors with images not being available yet because of the delay in the data being processed by Sage.
- the time slot configuration in get_sage.py is calculated based on the total number of days between the start and end date, and then choosing around 1 time slot per week, but capping to a reasonable number for distributed sampling.
    - Scales with time range: longer ranges get more slots
    - Ensures a minimum: at least 20 slots for short ranges
    - Caps the maximum: avoids too many queries
    - Distributes sampling: random slots across the range improve temporal diversity
- VSN list in get_sage.py is set to None to query all available VSNs, BUT the urban nodes are included so the dataset will need to be private.
    - two datasets will be created, one private and one public. The private dataset will include the urban nodes, but the public dataset will not.
- Note that the top camera images are removed from the dataset because they are not relevant to the benchmark. They generally show the sky with no objects in the frame.