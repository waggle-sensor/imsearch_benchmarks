# Sage Image Search Benchmarks

This repository contains the code and configuration for creating image retrieval benchmarks for Sage Image Search using the [imsearch_benchmaker](https://github.com/waggle-sensor/imsearch_benchmaker) framework. It also contains other datasets that we use to benchmark text-to-image retrieval systems in various scientific domains.

## Overview

This repository provides tools and pipelines to create standardized benchmark datasets for evaluating text-to-image retrieval systems in various scientific domains. Each benchmark follows a consistent pipeline architecture that automates the entire dataset creation process, from raw image collection to publication on Hugging Face. It also contains other datasets that we use to benchmark text-to-image retrieval systems in various scientific domains.

## Datasets

| Dataset | Domain | Description | Final Dataset | Code |
|---------|--------|-------------|--------------|--------------|
| FireBench | Fire Science 🔥 | A benchmark dataset for evaluating text-to-image retrieval systems in the domain of fire science. | [FireBench on Hugging Face](https://huggingface.co/datasets/sagecontinuum/FireBench) | [FireBenchMaker](FireBenchMaker/) |
| CommonObjectsBench | General Objects & Scenes 🌍 | A benchmark dataset for evaluating text-to-image retrieval systems on general objects and common scenes. | [CommonObjectsBench on Hugging Face](https://huggingface.co/datasets/sagecontinuum/CommonObjectsBench) | [CommonObjectsBenchMaker](CommonObjectsBenchMaker/) |
| Inquire | Biology 🌿 | A benchmark dataset for evaluating text-to-image retrieval systems in the domain of biology. | [INQUIRE-Benchmark-small on Hugging Face](https://huggingface.co/datasets/sagecontinuum/INQUIRE-Benchmark-small) | [Inquire](Inquire/) |

## Framework

All benchmarks in this repository use the **[imsearch_benchmaker](https://github.com/waggle-sensor/imsearch_benchmaker)** framework, which provides:

- Automated pipeline execution (preprocessing → annotation → query planning → judging → postprocessing)
- Integration with adapters for vision annotation and query generation (OpenAI, Google, etc.)
- Adapters for similarity scoring (apple/DFN5B-CLIP-ViT-H-14-378)
- Hugging Face dataset preparation and upload
- Exploratory data analysis tools

For detailed instructions, see the individual benchmark README files.

## Repository Structure

```
imsearch_benchmarks/
├── README.md                 # This file
├── FireBenchMaker/          # FireBench benchmark
│   ├── README.md            # FireBench documentation
│   ├── config.toml          # Pipeline configuration
│   ├── dataset_card.md      # Dataset card for Hugging Face
│   ├── requirements.txt     # Python dependencies
│   ├── tools/               # Data collection scripts
│   │   ├── get_figlib.py
│   │   ├── get_sage.py
│   │   └── get_wildfire.py
│   └── ...
└── ...
```

## Contributing

To add a new benchmark:

1. Create a new directory for your benchmark
2. Set up `config.toml` following the imsearch_benchmaker configuration format
3. Add data collection tools if needed
4. Create a README.md documenting your benchmark
5. If needed, add a new adapter for your benchmark
