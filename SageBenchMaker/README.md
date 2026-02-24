>TODO: update the README to conform to Sagebench
# SageBenchMaker

SageBenchMaker is a pipeline for creating **SageBench**, a benchmark dataset for evaluating text-to-image retrieval systems on **Sage Continuum** sensor images when queries reference **Sage metadata** (vsn, zone, host, job, plugin, camera, project, address). The tool uses the [imsearch_benchmaker](https://github.com/waggle-sensor/imsearch_benchmaker) framework to automate the process from raw image collection to Hugging Face dataset publication.

## Overview

### What is SageBench?

SageBench evaluates **image retrieval systems** on how well they retrieve Sage images when the query references Sage metadata (e.g., "images from node W097", "top camera in zone X"). Metadata is kept end-to-end: the data-fetching tool writes `metadata.jsonl` with Sage metadata fields, and the pipeline uses `vision_metadata_columns` and `user_prompt` in the config so that vision and judge stages see and use this context.

The dataset is designed to test retrieval systems' ability to:

- Retrieve images based on natural language queries that may reference Sage metadata (node/VSN, zone, host, job, plugin, camera, project, address)
- Combine visual relevance with metadata-aware matching when the query mentions metadata

**Final Public Dataset**: [SageBench on Hugging Face](https://huggingface.co/datasets/sagecontinuum/SageBench)  
**Final Private Dataset**: [SageBench-private on Hugging Face](https://huggingface.co/datasets/sagecontinuum/SageBench-private)

> **Note**: The private dataset may include imagery from urban Sage nodes, which are not allowed in the public dataset. Use the private dataset only in accordance with Sage Continuum and institutional policies; do not redistribute urban imagery publicly.

## Directory Structure

```
SageBenchMaker/
├── public/
│   ├── config.toml       # Config for public benchmark (non-urban nodes only)
│   └── dataset_card.md
├── private/
│   ├── config.toml       # Config for private benchmark
│   └── dataset_card.md
├── tools/
│   ├── get_sage.py       # Download Sage images and write metadata.jsonl
│   └── requirements.txt
├── README.md
├── env.template
├── requirements.txt
└── rights_map.json
```

The `public/` directory contains the configuration and dataset card for the public dataset (non-urban Sage nodes only).  
The `private/` directory contains the configuration and dataset card for the private dataset (may include urban nodes).  
The `tools/` directory contains the script used to collect images and write metadata.

## Source Data

All images come from **[Sage Continuum](https://sagecontinuum.org)**. The `tools/get_sage.py` script:

- Queries Sage for image data (configurable time frames, VSN filters)
- By default **excludes urban nodes**; set `SAGE_URBAN_IMAGERY=true` to include them (for the private benchmark)
- Downloads images to `image_root_dir/sage/`
- **Writes `metadata.jsonl`** at `image_root_dir/metadata.jsonl` with `image_id` plus Sage metadata: **vsn, zone, host, job, plugin, camera, project, address**

The benchmark uses this metadata in the vision and judge stages so that queries can reference it and relevance can consider both visual content and metadata match.

## Setup

1. **Environment** — Copy `env.template` to `.env` and set:

   - `OPENAI_API_KEY` — For vision and judge stages  
   - `HF_TOKEN` — For Hugging Face dataset upload  
   - `IMSEARCH_BENCHMAKER_CONFIG_PATH` — Path to **either** `public/config.toml` (public) **or** `private/config.toml` (private)  
   - `SAGE_USER` / `SAGE_PASSWORD` — For Sage Continuum image access  

   For the **private** benchmark (urban imagery), also set `SAGE_URBAN_IMAGERY=true` when running `get_sage.py`.

2. **Dependencies** — Install from `requirements.txt` (and `tools/requirements.txt` for the data-fetching tools).

3. **Paths** — Adjust paths in the chosen `public/config.toml` or `private/config.toml` (e.g., `image_root_dir`, `metadata_jsonl`, `images_jsonl`, `seeds_jsonl`, output dirs) to match your environment.

## Pipeline

1. **Fetch images and metadata** (one-time):

   ```bash
   cd tools && pip install -r requirements.txt && python get_sage.py
   ```

   - For **public**: use default (urban nodes excluded).  
   - For **private**: set `SAGE_URBAN_IMAGERY=true` before running to include urban nodes.

   This downloads images to `/tmp/SageBench/images/sage/` and writes `/tmp/SageBench/images/metadata.jsonl`.

2. **Run the imsearch_benchmaker pipeline** with the config for the variant you want:

   - **Public**: `export IMSEARCH_BENCHMAKER_CONFIG_PATH="/path/to/SageBenchMaker/public/config.toml"`
   - **Private**: `export IMSEARCH_BENCHMAKER_CONFIG_PATH="/path/to/SageBenchMaker/private/config.toml"`

   Then run:

   - `benchmaker preprocess` — Builds `images.jsonl`, `seeds.jsonl` (uses `metadata_jsonl` to attach Sage metadata to images)
   - `benchmaker annotate` — Vision annotation; prompts include `{metadata.vsn}`, `{metadata.zone}`, etc. via `vision_metadata_columns` and `user_prompt`
   - `benchmaker plan` — Query planning
   - `benchmaker judge` — Relevance labeling; judge prompt states that queries may reference Sage metadata
   - `benchmaker postprocess similarity` / `benchmaker postprocess summary` — Similarity and summary
   - `benchmaker upload` — Upload to Hugging Face (optional)

   See [imsearch_benchmaker](https://github.com/waggle-sensor/imsearch_benchmaker) for full pipeline documentation.

## Configuration

Key settings in `public/config.toml` and `private/config.toml`:

- **metadata_jsonl** — Path to the file written by `get_sage.py` (e.g. `/tmp/SageBench/images/metadata.jsonl`).
- **vision_metadata_columns** — `["vsn", "zone", "host", "job", "plugin", "camera", "project", "address"]`; these columns are merged from `metadata_jsonl` and passed into the vision and judge prompts.
- **vision_config.user_prompt** — Includes a “Metadata context” line that references `{metadata.vsn}`, `{metadata.zone}`, etc., so the model and downstream retrieval evaluation can use this context.
- **judge_config.user_prompt** — Instructs the judge that queries may reference Sage metadata and that relevance should consider both visual content and metadata match when the query mentions them.

**Public vs private**: The only functional differences between the two configs are `benchmark_name`, `_hf_repo_id`, `_hf_private`, and `hf_dataset_card_path`. Use the public config for the non-urban release and the private config for the variant that may include urban node imagery.

## Acknowledgments

We thank the creators and maintainers of Sage Continuum.
