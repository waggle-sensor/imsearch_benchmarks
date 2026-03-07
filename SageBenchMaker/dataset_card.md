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
    - name: environment_type
      dtype: float64
    - name: sky_condition
      dtype: float64
    - name: horizon_present
      dtype: float64
    - name: ground_present
      dtype: float64
    - name: sky_dominates
      dtype: float64
    - name: vegetation_present
      dtype: float64
    - name: water_present
      dtype: float64
    - name: buildings_present
      dtype: float64
    - name: vehicle_present
      dtype: float64
    - name: person_present
      dtype: float64
    - name: animal_present
      dtype: float64
    - name: night_scene
      dtype: float64
    - name: precipitation_visible
      dtype: float64
    - name: multiple_objects
      dtype: float64
  - name: viewpoint
    dtype: string
  - name: lighting
    dtype: string
  - name: environment_type
    dtype: string
  - name: sky_condition
    dtype: string
  - name: horizon_present
    dtype: bool
  - name: ground_present
    dtype: bool
  - name: sky_dominates
    dtype: bool
  - name: vegetation_present
    dtype: bool
  - name: water_present
    dtype: bool
  - name: buildings_present
    dtype: bool
  - name: vehicle_present
    dtype: bool
  - name: person_present
    dtype: bool
  - name: animal_present
    dtype: bool
  - name: night_scene
    dtype: bool
  - name: precipitation_visible
    dtype: bool
  - name: multiple_objects
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

SageBench is a benchmark dataset for evaluating **image retrieval systems** on **Sage Continuum** sensor network images when **queries reference Sage metadata** (vsn, zone, host, job, plugin, camera, project, address). The dataset consists of natural language queries paired with images and binary relevance labels. Metadata is retained end-to-end so that retrieval can be evaluated on both visual content and metadata-aware queries.

![Image Sample](summary/random_image_sample.png)

### Dataset Summary

SageBench contains:

- **Queries**: Natural language queries that **must reference at least one** Sage metadata field (e.g., node/VSN, camera, zone, job, project, address) and visual content
- **Images**: Sage Continuum sensor network imagery
- **Relevance Labels**: Binary labels (0 = not relevant, 1 = relevant) for each query–image pair
- **Sage metadata**: For each image: **vsn, zone, host, job, plugin, camera, project, address**. `project` and `address` are fetched from the Sage manifest API (not in query meta)
- **Annotations**: Summaries, tags, taxonomy (viewpoint, lighting, environment_type, sky_condition), and boolean scene flags
- **CLIPScore**: Pre-computed CLIP similarity scores using apple/DFN5B-CLIP-ViT-H-14-378

The dataset is designed to evaluate:

- Text-to-image retrieval when queries mention Sage metadata (node, camera, zone, job, project, address, etc.)
- Combination of visual relevance and metadata match

Image Proportions by categories:
![Image Proportions](summary/image_proportion_donuts.png)

### Supported Tasks and Leaderboards

- **Image Retrieval**: Given a text query that references sensor metadata and/or visual content, retrieve relevant images from a candidate pool
- **Relevance Classification**: Classify whether an image is relevant to a given query, including metadata match when the query references it
- **Multimodal Similarity**: Evaluate semantic similarity between text queries and images

### Languages

The dataset contains English text queries and image annotations.

## Dataset Structure

### Data Instances

Each instance includes the standard retrieval fields plus Sage metadata and annotations:

```python
{
    "query_id": "sagebench_q001",
    "query_text": "An animal in top camera of node W097",
    "image_id": "sage/...",
    "relevance_label": 1,
    "image": <PIL.Image.Image>,
    "license": "...",
    "doi": "...",
    "summary": "...",
    "tags": ["sky", "clouds", "animal", ...],
    "viewpoint": "ground_upward",
    "lighting": "day",
    "environment_type": "vegetation",
    "sky_condition": "clear",
    "horizon_present": false,
    "ground_present": true,
    "sky_dominates": false,
    "vegetation_present": true,
    "water_present": false,
    "buildings_present": false,
    "vehicle_present": false,
    "person_present": false,
    "animal_present": true,
    "night_scene": false,
    "precipitation_visible": false,
    "multiple_objects": true,
    "vsn": "W097",
    "zone": "core",
    "host": "000048b02d3ae2f2.ws-nxcore",
    "job": "imagesampler-bottom-2128",
    "plugin": "registry.sagecontinuum.org/yonghokim/imagesampler:0.3.4",
    "camera": "top",
    "project": "SAGE",
    "address": "Hawaii Volcanoes National Park, Pahoa, HI 96778",
    "confidence": { "viewpoint": 0.9, "lighting": 0.9, "environment_type": 0.9, "sky_condition": 0.9, ... },
    "clip_score": 2.56
}
```

### Data Fields

- **query_id** (string): Unique identifier for the query
- **query_text** (string): Natural language query; must reference at least one Sage metadata field (vsn, zone, host, job, plugin, camera, project, address) and visual content
- **image_id** (string): Unique identifier for the image (relative path, e.g. under `sage/`)
- **relevance_label** (int): Binary relevance (0 or 1)
- **image** (Image): The image file
- **license** (string): License information
- **doi** (string): Source dataset DOI
- **summary** (string): Brief factual summary (≤30 words)
- **tags** (list of strings): Controlled vocabulary tags (12–18 per image)
- **viewpoint** (string): Camera perspective (ground_upward, ground_horizontal, oblique, fisheye_sky, street_view, overhead, distant, duo_view, other, unknown)
- **lighting** (string): Lighting conditions (day, night, dusk, overcast_light, other, unknown)
- **environment_type** (string): What dominates the scene (sky_dominant, ground_dominant, mixed, urban, rural, vegetation, water, other, unknown)
- **sky_condition** (string): Weather/atmosphere (clear, partly_cloudy, overcast, fog_or_haze, precipitation, other, unknown)
- **horizon_present**, **ground_present**, **sky_dominates**, **vegetation_present**, **water_present**, **buildings_present**, **vehicle_present**, **person_present**, **animal_present**, **night_scene**, **precipitation_visible**, **multiple_objects** (bool): Boolean scene flags
- **vsn**, **zone**, **host**, **job**, **plugin**, **camera**, **project**, **address** (string): Sage metadata. `project` and `address` are obtained from the Sage manifest API per VSN
- **confidence** (dict): Confidence scores for viewpoint, lighting, environment_type, and sky_condition
- **clip_score** (float): Pre-computed CLIP similarity score

### Data Splits

The dataset is provided as a single split. Users can define train/validation/test splits as needed.

## Dataset Creation

### Curation Rationale

SageBench was created to evaluate image retrieval when **queries reference Sage metadata**. Goals:

1. **Metadata-aware retrieval**: Queries must reference at least one metadata field (vsn, zone, host, job, plugin, camera, project, address); relevance depends on both visual content and metadata match.
2. **Sage-only imagery**: All images from the Sage Continuum sensor network (national network, 100+ nodes, 17 states; RGB/IR cameras, LiDAR, environmental sensors).
3. **Retained metadata**: The pipeline keeps all eight metadata fields and exposes them via `vision_metadata_columns` and `user_prompt`. `project` and `address` are fetched from the manifest API in `tools/get_sage.py` (they are not in the query meta).

### Source Data

- **Sage Continuum** ([Sage](https://sagecontinuum.org))  
  - National AI infrastructure with edge computing and multimodal sensing  
  - DOI: 10.1109/ICSENS.2016.7808975  

Images are fetched with `tools/get_sage.py`, which queries Sage, downloads images to `image_root_dir/sage/`, and writes `metadata.jsonl` with **vsn, zone, host, job, plugin, camera** from the query response and **project, address** from the Sage manifest API (`https://auth.sagecontinuum.org/manifests/<VSN>`). The imsearch_benchmaker pipeline merges this metadata and uses it in vision and judge prompts.

### Annotations

1. **Vision annotation**: OpenAI vision API (gpt-5-mini); user prompt includes sensor metadata context and taxonomy (viewpoint, lighting, environment_type, sky_condition) and boolean fields.
2. **Query generation**: Queries must reference at least one sensor metadata field and visual content.
3. **Relevance labeling**: AI judge (gpt-5-mini); judge prompt requires metadata-aware queries and considers both visual content and metadata match.
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

The dataset was created using imsearch_benchmaker. See [imsearch_benchmarks/SageBenchMaker](https://github.com/waggle-sensor/imsearch_benchmarks/tree/main/SageBenchMaker) for the code and configuration. Use `config.toml` for the benchmark; **vision_metadata_columns** and **user_prompt** define how Sage metadata is included in the pipeline.

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
	doi          = { FILL_IN_DOI }, #TODO: Add DOI
	publisher    = { Hugging Face }
}
```

## Dataset Statistics

Please refer to the [EDA](summary/SageBench_eda_analysis.ipynb) in the [summary/](summary/) directory.

## Hyperparameters in creating the dataset

Please refer to the [config_values.csv](summary/config_values.csv) file in the `summary/` directory for the values of the hyperparameters used in the dataset creation.

| value | description |
|-------|-------------|
query_plan_num_seeds | the number of seed images to use for query generation
query_plan_pos_total | the number of positive images to generate for each query
query_plan_neutral_total | the number of neutral images to generate for each query
query_plan_neg_total | the total number of negatives to generate for each query
query_plan_neg_hard | the number of hard negatives to generate for each query
query_plan_neg_easy | the number of easy negatives to generate for each query
query_plan_random_seed | the random seed used for reproducibility
query_plan_seed_image_ids_column | the column name for seed image IDs
query_plan_candidate_image_ids_column | the column name for candidate image IDs
columns_boolean | the list of boolean scene flags (horizon_present, ground_present, sky_dominates, etc.)
columns_taxonomy | the taxonomy dimensions and allowed values (viewpoint, lighting, environment_type, sky_condition)
controlled_tag_vocab | the controlled tag vocabulary for the SageBench benchmark
min_tags | minimum number of tags per image
max_tags | maximum number of tags per image
vision_config.adapter | the adapter for the vision annotation (e.g. openai)
vision_config.model | the model for the vision annotation (e.g. gpt-5-mini)
vision_config.system_prompt | the system prompt for the vision annotation
vision_config.user_prompt | the user prompt for the vision annotation
vision_config.max_output_tokens | the maximum number of tokens for the vision annotation
vision_config.reasoning_effort | the reasoning effort for the vision annotation
vision_config.image_detail | the image detail level (low, medium, high)
vision_config.max_images_per_batch | the maximum number of images per vision batch shard
vision_config.completion_window | the completion window for the batch
vision_config.vision_metadata_columns | the Sage metadata columns included in the vision annotation (vsn, zone, host, job, plugin, camera, project, address)
vision_config.price_per_million_input_tokens | the price per million input tokens for the vision annotation
vision_config.price_per_million_output_tokens | the price per million output tokens for the vision annotation
vision_config.price_per_million_cached_input_tokens | the price per million cached input tokens for the batch
vision_config.price_per_million_image_input_tokens | the price per million image input tokens for the batch
vision_config.price_per_million_image_output_tokens | the price per million image output tokens for the batch
judge_config.adapter | the adapter for the judge (e.g. openai)
judge_config.model | the model for the judge (e.g. gpt-5-mini)
judge_config.system_prompt | the system prompt for the judge
judge_config.user_prompt | the user prompt for the judge
judge_config.max_output_tokens | the maximum number of tokens for the judge
judge_config.reasoning_effort | the reasoning effort for the judge
judge_config.max_queries_per_batch | the maximum number of queries per judge batch shard
judge_config.max_candidates | the maximum number of candidates per query
judge_config.completion_window | the completion window for the batch
judge_config.price_per_million_input_tokens | the price per million input tokens for the judge
judge_config.price_per_million_output_tokens | the price per million output tokens for the judge
judge_config.price_per_million_cached_input_tokens | the price per million cached input tokens for the judge
similarity_config.adapter | the adapter for the similarity scoring (e.g. local_clip)
similarity_config.model | the model for the similarity scoring (e.g. apple/DFN5B-CLIP-ViT-H-14-378)
similarity_config.col_name | the column name for the similarity score
similarity_config.device | the device to run the similarity scoring on
similarity_config.use_safetensors | whether to use safetensors for the similarity scoring

### Acknowledgments

We thank the creators and maintainers of Sage Continuum.

## References

```
Catlett, C. E., P. H. Beckman, R. Sankaran, and K. K. Galvin, 2017: Array of Things: A Scientific Research Instrument in the Public Way: Platform Design and Early Lessons Learned. Proceedings of the 2nd International Workshop on Science of Smart City Operations and Platforms Engineering, 26–33. https://doi.org/10.1109/ICSENS.2016.7808975
```
