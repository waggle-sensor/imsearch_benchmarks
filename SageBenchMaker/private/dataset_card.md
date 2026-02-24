---
dataset_info:
  features:
  - name: query_id
    dtype: string
  - name: query_text
    dtype: string
  - name: image_id
    dtype: string
  - name: relevance_label
    dtype: int64
  - name: doi
    dtype: string
  - name: license
    dtype: string
  - name: summary
    dtype: string
  - name: tags
    sequence: string
  - name: clip_score
    dtype: float64
  - name: image
    dtype: image
  - name: confidence
    struct:
    - name: viewpoint
      dtype: float64
    - name: lighting
      dtype: float64
  - name: viewpoint
    dtype: string
  - name: lighting
    dtype: string
  - name: horizon_visible
    dtype: bool
  - name: ground_visible
    dtype: bool
  - name: sky_dominates
    dtype: bool
  - name: vsn
    dtype: string
  - name: zone
    dtype: string
  - name: host
    dtype: string
  - name: job
    dtype: string
  - name: plugin
    dtype: string
  - name: camera
    dtype: string
  - name: project
    dtype: string
  - name: address
    dtype: string
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
license: cc-by-4.0
task_categories:
- image-classification
language:
- en
tags:
- image-retrieval
- benchmark
- computer-vision
- sage-continuum
- metadata-aware
size_categories:
- 1K<n<10K
pretty_name: Sage Benchmark (Private)
---

# SageBench-private: A Benchmark for Sage Image Retrieval with Metadata-Aware Queries

## Dataset Description

SageBench-private is the **private** variant of SageBench, a benchmark for evaluating **image retrieval systems** on **Sage Continuum** sensor images when **queries reference Sage metadata** (vsn, zone, host, job, plugin, camera, project, address). The dataset consists of natural language queries paired with images and binary relevance labels. Metadata is retained end-to-end so that retrieval can be evaluated on both visual content and metadata-aware queries.

**Private variant**: This dataset **may include imagery from urban Sage nodes**, which are not permitted in the public SageBench release. This variant is hosted as a private Hugging Face dataset. Please use it only in accordance with Sage Continuum and dataset terms; do not redistribute urban imagery publicly.

### Dataset Summary

SageBench-private contains:

- **Queries**: Natural language queries that may reference Sage metadata (e.g., node/VSN, camera, zone, job) and/or visual content
- **Images**: Sage Continuum sensor network imagery (may include urban nodes)
- **Relevance Labels**: Binary labels (0 = not relevant, 1 = relevant) for each query–image pair
- **Sage metadata**: For each image, metadata is retained: **vsn, zone, host, job, plugin, camera, project, address** for query-reference evaluation
- **Annotations**: Summaries, tags, viewpoint, lighting, and boolean flags (horizon_visible, ground_visible, sky_dominates)
- **CLIPScore**: Pre-computed CLIP similarity scores using apple/DFN5B-CLIP-ViT-H-14-378

The dataset is designed to evaluate:

- Text-to-image retrieval when queries mention Sage metadata (node, camera, zone, etc.)
- Combination of visual relevance and metadata match

### Supported Tasks and Leaderboards

- **Image Retrieval**: Given a text query (possibly referencing metadata), retrieve relevant images from a candidate pool
- **Relevance Classification**: Classify whether an image is relevant to a given query, including metadata match when the query references it
- **Multimodal Similarity**: Evaluate semantic similarity between text queries and images

### Languages

The dataset contains English text queries and image annotations.

## Dataset Structure

### Data Instances

Each instance includes the standard retrieval fields plus Sage metadata when present (same schema as SageBench public). See the public dataset card for full field descriptions.

### Data Splits

The dataset is provided as a single split. Users can define train/validation/test splits as needed.

## Dataset Creation

### Curation Rationale

SageBench-private was created to evaluate image retrieval when **queries reference Sage metadata**, with the option to **include urban Sage node imagery** for internal or restricted use. Goals match the public benchmark; the difference is the allowed node set (urban nodes included when data is collected with `SAGE_URBAN_IMAGERY=true`).

### Source Data

- **Sage Continuum** ([Sage](https://sagecontinuum.org))  
  - Cyberinfrastructure sensor network imagery; this private variant may include urban nodes  
  - DOI: 10.1109/ICSENS.2016.7808975  

Images are fetched with `tools/get_sage.py`. To include urban nodes, set the environment variable `SAGE_URBAN_IMAGERY=true` before running the script. The script writes `metadata.jsonl` with the eight Sage metadata fields per image. The imsearch_benchmaker pipeline is run with `private/config.toml`.

### Annotations

Same annotation pipeline as the public benchmark (vision annotation, query generation, relevance labeling, CLIPScore). Use `private/config.toml` for the pipeline configuration.

### Personal and Sensitive Information

The dataset uses Sage Continuum imagery. Urban node imagery may be subject to additional use restrictions. No personal information is included.

## Considerations for Using the Data

### Social Impact

Supports research and development of metadata-aware image retrieval and multimodal systems over sensor network imagery. **Use the private dataset only in line with Sage Continuum and institutional policies; do not redistribute urban imagery publicly.**

### Other Known Limitations

- Queries are model-generated and may not cover all real-world metadata-reference patterns.
- Binary relevance may not capture graded relevance.
- Image set is limited to Sage nodes and time ranges used in data collection.

## Additional Information

### Dataset Curators

The dataset was created using imsearch_benchmaker. See [imsearch_benchmarks/SageBenchMaker](https://github.com/waggle-sensor/imsearch_benchmarks/tree/main/SageBenchMaker) for the code and configuration. Use `private/config.toml` for this private benchmark. **vision_metadata_columns** and **user_prompt** define how Sage metadata (vsn, zone, host, job, plugin, camera, project, address) is included in the pipeline.

### Licensing Information

Sage Continuum imagery is used according to source terms. Urban imagery may have additional restrictions. The dataset card and annotations are provided under CC BY 4.0.

### Citation Information

If you use this dataset, please cite:

```bibtex
@misc{sagebench_private_2026,
	author       = { Francisco Lozano },
	affiliation  = { Northwestern University },
	title        = { SageBench-private },
	year         = 2026,
	url          = { https://huggingface.co/datasets/sagecontinuum/SageBench-private },
	publisher    = { Hugging Face }
}
```

### Acknowledgments

We thank the creators and maintainers of Sage Continuum.

## References

```
Catlett, C. E., P. H. Beckman, R. Sankaran, and K. K. Galvin, 2017: Array of Things: A Scientific Research Instrument in the Public Way: Platform Design and Early Lessons Learned. Proceedings of the 2nd International Workshop on Science of Smart City Operations and Platforms Engineering, 26–33. https://doi.org/10.1109/ICSENS.2016.7808975
```
