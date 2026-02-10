# CloudBenchMaker

CloudBenchMaker is a pipeline for creating **CloudBench**, a benchmark dataset for evaluating text-to-image retrieval systems in the domain of Atmospheric Science, with a focus on clouds. The tool uses the [imsearch_benchmaker](https://github.com/waggle-sensor/imsearch_benchmaker) framework to automate the entire dataset creation process, from raw image collection to Hugging Face dataset publication.

## Overview

### What is CloudBench?

CloudBench is a benchmark dataset for evaluating **text-to-image retrieval systems** in the domain of Atmospheric Science, specifically focused on clouds. Given a natural language query (e.g., "A cloud in the sky"), the system should retrieve relevant images from a candidate pool.

The dataset is designed to test retrieval systems' ability to:

- Find relevant cloud-related images based on natural language queries
- Handle diverse cloud types, sizes, and positions
- Support atmospheric science research and early detection systems

**Final Dataset**: [CloudBench on Hugging Face](https://huggingface.co/datasets/sagecontinuum/CloudBench)

## Directory Structure
```
CloudBenchMaker/
├── tools/
├── config.toml
├── README.md
├── dataset_card.md
├── env.template
└── requirements.txt
└──rights_map.json
```

The `tools/` directory contains the tools used to collect the images.
The `config.toml` file contains the configuration for the pipeline.
The `README.md` file contains the README for the repository.
The `dataset_card.md` file contains the dataset card for the dataset.
The `env.template` file contains the environment template for the pipeline.
The `requirements.txt` file contains the requirements for the pipeline.
The `rights_map.json` file contains the rights map for the dataset.

## Source Datasets

1. **[CCSN](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CADDPD)** — Cirrus Cumulus Stratus Nimbus dataset  
2. **[Sage Continuum](https://sagecontinuum.org)** — Cyberinfrastructure sensor network imagery

### Dataset Composition

The pipeline is configured for a **50/50 split** between CCSN and Sage images:

- **CCSN (50%)** — Provides clear cloud images and greater diversity  
- **Sage (50%)** — Provides real-world imagery with less diversity  

The sample size for both source datasets is **2,543 images** each, so the initial source split is 50% from each. The final dataset composition can differ after query planning, since the framework selects candidates based on Jaccard similarity and other criteria.

Some Sage images are not cloud-relevant (e.g., empty sky, non-cloud scenes). Keeping them in the dataset better reflects real-world Sage node imagery and tests the retrieval system’s ability to handle irrelevant candidates; they can be manually filtered if desired.

## Setup

1. **Environment** — Copy `env.template` to `.env` and set:

   - `OPENAI_API_KEY` — For vision and judge stages  
   - `HF_TOKEN` — For Hugging Face dataset upload  
   - `IMSEARCH_BENCHMAKER_CONFIG_PATH` — Path to this repo’s `config.toml`  
   - `SAGE_USER` / `SAGE_PASSWORD` — For Sage Continuum image access (if applicable)

2. **Dependencies** — Install from `requirements.txt` (and `tools/requirements.txt` if using the data-fetching tools).

3. **Paths** — Adjust paths in `config.toml` (e.g., `image_root_dir`, `images_jsonl`, `seeds_jsonl`, output dirs) to match your environment.

## Pipeline Notes & Known Issues

These notes come from running the full CloudBench creation pipeline and may be useful when reproducing or extending the dataset:

- **Vision stage** — **623 images** were lost due to URL timeouts when fetching images for annotation. Retries led to some images being annotated multiple times, which increased cost in this stage.
- **Judge stage** — **5 queries** were dropped when the judge model hit the max-tokens limit.
- **Similarity scoring** — **68 image–query pairs** were lost due to unknown errors in the similarity scoring process.

When reproducing, expect some attrition at each stage; the exact numbers may vary with network conditions, API limits, and framework versions.

## Acknowledgments

We thank the creators and maintainers of:

- CCSN Dataset  
- Sage Continuum  
