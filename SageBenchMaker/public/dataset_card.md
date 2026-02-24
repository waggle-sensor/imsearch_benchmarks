---
#TODO: update the dataset card to conform to Sagebench
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
pretty_name: Sage Benchmark
---

# SageBench: A Benchmark for Sage Image Retrieval with Metadata-Aware Queries

## Dataset Description

SageBench is a benchmark dataset for evaluating **image retrieval systems** on **Sage Continuum** sensor images when **queries reference Sage metadata** (vsn, zone, host, job, plugin, camera, project, address). The dataset consists of natural language queries paired with images and binary relevance labels. Metadata is retained end-to-end so that retrieval can be evaluated on both visual content and metadata-aware queries.

### Dataset Summary

SageBench contains:

- **Queries**: Natural language queries that may reference Sage metadata (e.g., node/VSN, camera, zone, job) and/or visual content
- **Images**: Sage Continuum sensor network imagery (non-urban nodes)
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

Each instance includes the standard retrieval fields plus Sage metadata when present:

```python
{
    "query_id": "sagebench_q001",
    "query_text": "Images from node W097 with top camera",
    "image_id": "sage/...",
    "relevance_label": 1,
    "image": <PIL.Image.Image>,
    "license": "...",
    "doi": "...",
    "summary": "...",
    "tags": ["sky", "clouds", ...],
    "viewpoint": "ground_upward",
    "lighting": "day",
    "horizon_visible": false,
    "ground_visible": false,
    "sky_dominates": true,
    "vsn": "W097",
    "zone": "...",
    "host": "...",
    "job": "...",
    "plugin": "...",
    "camera": "top",
    "project": "...",
    "address": "...",
    "confidence": { "viewpoint": 0.9, "lighting": 0.9 },
    "clip_score": ...
}
```

### Data Fields

- **query_id** (string): Unique identifier for the query
- **query_text** (string): Natural language query; may reference Sage metadata (vsn, zone, host, job, plugin, camera, project, address) and/or visual content
- **image_id** (string): Unique identifier for the image (relative path, e.g. under `sage/`)
- **relevance_label** (int): Binary relevance (0 or 1)
- **image** (Image): The image file
- **license** (string): License information
- **doi** (string): Source dataset DOI
- **summary** (string): Brief factual summary (≤30 words)
- **tags** (list of strings): Controlled vocabulary tags (12–18 per image)
- **viewpoint** (string): Camera perspective (e.g. ground_upward, oblique, unknown)
- **lighting** (string): Lighting conditions (day, night, dusk, other, unknown)
- **horizon_visible**, **ground_visible**, **sky_dominates** (bool): Boolean scene flags
- **vsn**, **zone**, **host**, **job**, **plugin**, **camera**, **project**, **address** (string): Sage metadata retained for query-reference evaluation
- **confidence** (dict): Confidence scores for viewpoint and lighting
- **clip_score** (float): Pre-computed CLIP similarity score

### Data Splits

The dataset is provided as a single split. Users can define train/validation/test splits as needed.

## Dataset Creation

### Curation Rationale

SageBench was created to evaluate image retrieval when **queries reference Sage metadata**. Goals:

1. **Metadata-aware retrieval**: Queries may mention node (VSN), zone, host, job, plugin, camera, project, or address; relevance can depend on both visual content and metadata match.
2. **Sage-only imagery**: All images from Sage Continuum sensor network.
3. **Retained metadata**: The pipeline keeps **vsn, zone, host, job, plugin, camera, project, address** and exposes them via `vision_metadata_columns` and `user_prompt` in the benchmark config so that vision and judge stages use this context.

### Source Data

- **Sage Continuum** ([Sage](https://sagecontinuum.org))  
  - Cyberinfrastructure sensor network imagery
  - DOI: 10.1109/ICSENS.2016.7808975  

Images are fetched with `tools/get_sage.py` (without urban nodes). The script also writes `metadata.jsonl` with the eight Sage metadata fields per image. The imsearch_benchmaker pipeline merges this metadata and uses it in vision and judge prompts.

### Annotations

1. **Vision annotation**: OpenAI vision API (gpt-5-mini); user prompt includes metadata context (`{metadata.vsn}`, `{metadata.zone}`, etc.) via `vision_metadata_columns`.
2. **Query generation**: Queries may reference Sage metadata or visual content.
3. **Relevance labeling**: AI judge (gpt-5-mini); judge prompt states that queries may reference Sage metadata and that relevance should consider both visual content and metadata match.
4. **CLIPScore**: Local CLIP model (apple/DFN5B-CLIP-ViT-H-14-378).

### Personal and Sensitive Information

The dataset uses Sage Continuum imagery. No personal information is included.

## Considerations for Using the Data

### Social Impact

Supports research and development of metadata-aware image retrieval and multimodal systems over sensor network imagery.

### Other Known Limitations

- Queries are model-generated and may not cover all real-world metadata-reference patterns.
- Binary relevance may not capture graded relevance.
- Image set is limited to Sage nodes and time ranges used in data collection.

## Additional Information

### Dataset Curators

The dataset was created using imsearch_benchmaker. See [imsearch_benchmarks/SageBenchMaker](https://github.com/waggle-sensor/imsearch_benchmarks/tree/main/SageBenchMaker) for the code and configuration. **vision_metadata_columns** and **user_prompt** define how Sage metadata (vsn, zone, host, job, plugin, camera, project, address) is included in the pipeline.

### Licensing Information

Sage Continuum imagery is used according to source terms. The dataset card and annotations are provided under CC BY 4.0.

### Citation Information

If you use this dataset, please cite:

```bibtex
@misc{sagebench_2026,
	author       = { Francisco Lozano },
	affiliation  = { Northwestern University },
	title        = { SageBench },
	year         = 2026,
	url          = { https://huggingface.co/datasets/sagecontinuum/SageBench },
	publisher    = { Hugging Face }
}
```

### Acknowledgments

We thank the creators and maintainers of Sage Continuum.

## References

```
Catlett, C. E., P. H. Beckman, R. Sankaran, and K. K. Galvin, 2017: Array of Things: A Scientific Research Instrument in the Public Way: Platform Design and Early Lessons Learned. Proceedings of the 2nd International Workshop on Science of Smart City Operations and Platforms Engineering, 26–33. https://doi.org/10.1109/ICSENS.2016.7808975
```
