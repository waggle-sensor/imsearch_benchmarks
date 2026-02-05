# CloudBenchMaker

CloudBenchMaker is a pipeline for creating **CloudBench**, a benchmark dataset for evaluating text-to-image retrieval systems in the domain of Atmospheric Science specifically focused on clouds. The tool uses the [imsearch_benchmaker](https://github.com/waggle-sensor/imsearch_benchmaker) framework to automate the entire dataset creation process, from raw image collection to Hugging Face dataset publication.

## Overview

### What is CloudBench?

CloudBench is a benchmark dataset for evaluating **text-to-image retrieval systems** in the domain of Atmospheric Science specifically focused on clouds. Given a natural language query like "A cloud in the sky", the system should retrieve relevant images from a candidate pool.

The dataset is designed to test retrieval systems' ability to:
- Find relevant cloud-related images based on natural language queries
- Handle diverse cloud types, sizes, and positions
- Support atmospheric science research and early detection systems

## Source Datasets
1. **[CCSN](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CADDPD)**: Cirrus Cumulus Stratus Nimbus dataset
2. **[Sage Continuum](https://sagecontinuum.org)**: Cyberinfrastructure sensor network imagery

## Acknowledgments

We thank the creators and maintainers of:
- CCSN Dataset
- Sage Continuum