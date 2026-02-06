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
    - name: occlusion_present
      dtype: float64
    - name: confounder_type
      dtype: float64
    - name: cloud_coverage
      dtype: float64
    - name: viewpoint
      dtype: float64
    - name: lighting
      dtype: float64
    - name: environment_type
      dtype: float64
    - name: multiple_cloud_types
      dtype: float64
    - name: horizon_visible
      dtype: float64
    - name: ground_visible
      dtype: float64
    - name: sun_visible
      dtype: float64
    - name: precipitation_visible
      dtype: float64
    - name: overcast
      dtype: float64
    - name: multiple_layers
      dtype: float64
    - name: storm_visible
      dtype: float64
  - name: occlusion_present
    dtype: bool
  - name: confounder_type
    dtype: 
      class_label:
        names:
          - none
          - fog
          - haze
          - dust
          - smoke
          - sun_glare
          - precipitation
          - marine_layer
          - industrial_plume
          - multiple
          - unknown
  - name: cloud_coverage
    dtype: 
      class_label:
        names:
          - 0%-25%
          - 25%-50%
          - 50%-75%
          - 75%-100%
          - unknown
  - name: viewpoint
    dtype: 
      class_label:
        names:
          - ground_upward
          - ground_horizontal
          - fisheye_sky
          - oblique
          - other
          - unknown
  - name: lighting
    dtype: 
      class_label:
        names:
          - day
          - night
          - dusk
          - bright
          - overcast_light
          - other
          - unknown
  - name: environment_type
    dtype: 
      class_label:
        names:
          - forest
          - grassland
          - shrubland
          - mountainous
          - urban_wui
          - coastal
  - name: cloud_category
    dtype:
      class_label:
        names:
          - cirrus
          - cirrostratus
          - cirrocumulus
          - altocumulus
          - altostratus
          - cumulus
          - cumulonimbus
          - nimbostratus
          - stratocumulus
          - stratus
          - contrail
  - name: multiple_cloud_types
    dtype: bool
  - name: horizon_visible
    dtype: bool
  - name: ground_visible
    dtype: bool
  - name: sun_visible
    dtype: bool
  - name: precipitation_visible
    dtype: bool
  - name: overcast
    dtype: bool
  - name: multiple_layers
    dtype: bool
  - name: storm_visible
    dtype: bool
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
- clouds
- atmospheric-science
size_categories:
- 1K<n<10K
pretty_name: Cloud Benchmark
---

# CloudBench: A Benchmark Dataset for Cloud Image Retrieval

## Dataset Description

CloudBench is a benchmark dataset for evaluating image retrieval systems in the domain of Atmospheric Science specifically focused on clouds. The dataset consists of natural language queries paired with images, along with binary relevance labels indicating whether each image is relevant to the query. The dataset is designed to test retrieval systems' ability to find relevant images based on queries describing clouds.

![Image Sample](summary/random_image_sample.png)

### Dataset Summary

CloudBench contains:
- **Queries**: Natural language queries describing clouds and their properties
- **Images**: Real-world imagery from multiple sources (Cirrus Cumulus Stratus Nimbus (CCSN) dataset)
- **Relevance Labels**: Binary labels (0 = not relevant, 1 = relevant) for each query-image pair
- **Rich Metadata**: Comprehensive annotations to support both retrieval evaluation and analysis.
- **CLIPScore**: Pre-computed CLIP similarity scores for each query-image pair using apple/DFN5B-CLIP-ViT-H-14-378 model.

The dataset is designed to evaluate:
- Text-to-image retrieval systems
- General-purpose image search engines
- Multimodal understanding models

Image Proportions by categories:
![Image Proportions](summary/image_proportion_donuts.png)

### Supported Tasks and Leaderboards

- **Image Retrieval**: Given a text query, retrieve relevant images from a candidate pool
- **Relevance Classification**: Classify whether an image is relevant to a given query
- **Multimodal Similarity**: Evaluate semantic similarity between text queries and images

### Languages

The dataset contains English text queries and image annotations.

## Dataset Structure

### Data Instances

Each instance in the dataset contains:

```python
{
    "query_id": "cloudbench_q001",
    "query_text": "A cumulus cloud in the sky",
    "image_id": "ccsn/train/000000000000.jpg",
    "relevance_label": 1,
    "image": <PIL.Image.Image>,  # The actual image
    "license": "CC0 1.0",
    "doi": "10.7910/DVN/CADDPD",
    "tags": ["cumulus", ...],
    "occlusion_present": false,
    "confounder_type": "none",
    "cloud_coverage": "0%-25%",
    "viewpoint": "ground_upward",
    "lighting": "day",
    "environment_type": "forest",
    "multiple_cloud_types": false,
    "horizon_visible": false,
    "ground_visible": false,
    "sun_visible": false,
    "precipitation_visible": false,
    "overcast": false,
    "multiple_layers": false,
    "storm_visible": false,
    "confidence": {
        "occlusion_present": 0.9,
        "confounder_type": 0.9,
        "cloud_coverage": 0.9,
        "viewpoint": 0.9,
        "lighting": 0.9,
        "environment_type": 0.9,
        "multiple_cloud_types": 0.9,
        "horizon_visible": 0.9,
        "ground_visible": 0.9,
    },
    "summary": "A cumulus cloud in the sky.",
    "clip_score": 5.337447166442871
}
```

### Data Fields

- **query_id** (string): Unique identifier for the query (e.g., "cloudbench_q001")
- **query_text** (string): Natural language query describing the target cloud type, atmospheric condition, or meteorological feature
- **image_id** (string): Unique identifier for the image (relative path from source, e.g., "ccsn/Ci/image.jpg")
- **relevance_label** (int): Binary relevance label (0 = not relevant, 1 = relevant)
- **image** (Image): The actual image file
- **license** (string): License information for the image (e.g., "CC0 1.0")
- **doi** (string): Digital Object Identifier for the source dataset
- **cloud_category** (string): Cloud type classification from source metadata (e.g., "cirrus", "cumulus", "stratus", "cirrostratus", "cirrocumulus", "altocumulus", "altostratus", "cumulonimbus", "nimbostratus", "stratocumulus", "contrail")
- **viewpoint** (string): Camera perspective. Values: "ground_upward" (ground-based looking upward), "ground_horizontal" (ground-based horizontal), "fisheye_sky" (fisheye capturing full sky dome), "oblique" (angled view), "other", "unknown"
- **lighting** (string): Lighting conditions. Values: "day" (sunlit/daylight), "night" (low light/dark), "dusk" (twilight/sunset/sunrise), "bright" (high brightness/strong sunlight), "overcast_light" (overcast but daylight), "other", "unknown"
- **cloud_coverage** (string): Percentage of sky covered by clouds. Values: "0%-25%" (clear sky), "25%-50%" (scattered clouds), "50%-75%" (broken clouds), "75%-100%" (overcast), "unknown"
- **confounder_type** (string): Atmospheric phenomena that might obscure or be confused with clouds. Values: "none" (no confounders), "fog", "haze", "dust", "smoke", "sun_glare", "precipitation", "marine_layer", "industrial_plume", "multiple" (multiple confounders), "unknown"
- **occlusion_present** (bool): True if the cloud is partially occluded by other objects or clouds
- **multiple_cloud_types** (bool): True if multiple distinct cloud types are visible in the image
- **horizon_visible** (bool): True if the horizon line is visible in the image
- **ground_visible** (bool): True if ground or terrain is visible in the image
- **sun_visible** (bool): True if the sun is visible in the image
- **precipitation_visible** (bool): True if rain, snow, or other precipitation is visible in the image
- **overcast** (bool): True if the sky is completely overcast (no clear sky visible)
- **multiple_layers** (bool): True if multiple cloud layers are visible at different altitudes
- **storm_visible** (bool): True if a storm is visible (e.g., cumulonimbus with storm features)
- **tags** (list of strings): Controlled vocabulary tags describing atmospheric features, cloud characteristics, and scene elements (12-18 tags per image)
- **confidence** (dict): Confidence scores (0.0-1.0) for taxonomy fields: `viewpoint`, `lighting`, `cloud_coverage`, `confounder_type`. Higher values indicate higher certainty.
- **summary** (string): Brief factual summary of the cloud(s) and atmospheric conditions (≤30 words)
- **clip_score** (float): Pre-computed CLIP similarity score between query and image using apple/DFN5B-CLIP-ViT-H-14-378 model

### Data Splits

The dataset is provided as a single split. Users can create their own train/validation/test splits as needed.

## Dataset Creation

### Curation Rationale

CloudBench was created to address the need for a standardized benchmark for evaluating image retrieval systems in the domain of Atmospheric Science specifically focused on clouds. The dataset focuses on:

1. **Real-world diversity**: Images from multiple sources covering various clouds, sizes, and positions
2. **General-purpose queries**: Queries describing clouds and their properties
3. **Comprehensive annotations**: Rich metadata to support both retrieval evaluation and analysis

### Source Data

The dataset combines images from two sources:

1. **Cirrus Cumulus Stratus Nimbus (CCSN) Dataset** ([CCSN](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CADDPD))
   - Large-scale cloud imagery dataset
   - 2543 cloud images
   - 11 cloud types
   - DOI: 10.7910/DVN/CADDPD

2. **Sage Continuum** ([Sage](https://sagecontinuum.org))
   - Cyberinfrastructure sensor network imagery
   - DOI: 10.1109/ICSENS.2016.7808975

![Source proportion](summary/dataset_proportion_donuts.png)

### Annotations

#### Annotation process

1. **Vision Annotation**: Images were annotated using OpenAI's vision API (GPT-5-mini) to extract:
   - Image summaries
   - Controlled vocabulary tags (12-18 tags per image)
   - Confidence scores for annotations

2. **Query Generation**: Queries were generated using OpenAI's text API (GPT-5-mini) based on seed images

3. **Relevance Labeling**: Binary relevance labels were assigned by AI judges using OpenAI's text API (GPT-5-mini) based on query-image pairs

4. **CLIPScore Calculation**: CLIP similarity scores were computed using a local CLIP model (apple/DFN5B-CLIP-ViT-H-14-378)

>NOTE: You can find the annotations in the [annotations.jsonl](annotations.jsonl) file. Keep in mind that not all images are in the final dataset due to the query planning process, so you will see images in the annotations.jsonl file that are not in the final dataset.

#### Who are the annotators?

- **Vision annotations**: Automated using OpenAI's vision API
- **Queries**: Generated by OpenAI's text API
- **Relevance labels**: Assigned by OpenAI's text API

### Personal and Sensitive Information

The dataset contains publicly available imagery. No personal information is included.

## Considerations for Using the Data

### Social Impact of Dataset

This dataset supports research and development of:
- General-purpose image retrieval systems
- Multimodal understanding models
- Image search applications

### Discussion of Biases

Potential biases in the dataset:
- **Geographic bias**: Images may be biased toward regions where source datasets were collected
- **Temporal bias**: Images may reflect specific time periods when data was collected
- **Object bias**: Certain objects may be overrepresented based on source dataset composition

### Other Known Limitations

1. **Query diversity**: Queries are generated by AI models and may not fully capture all real-world search scenarios
2. **Relevance judgments**: Binary relevance labels may not capture nuanced relevance levels
3. **Image quality**: Images vary in resolution and quality depending on source

## Additional Information

### Dataset Curators

The dataset was created using imsearch_benchmaker, a pipeline for creating image retrieval benchmarks. See [imsearch_benchmarks/CloudBenchMaker](https://github.com/waggle-sensor/imsearch_benchmarks/tree/main/CloudBenchMaker) for the code implementation.

### Licensing Information

Images in the dataset are licensed according to their source:
- **CCSN Dataset**: See source for licensing information
- **Sage Continuum**: See source for licensing information

The dataset card and annotations are provided under CC BY 4.0.

### Citation Information

If you use this dataset, please cite:

```bibtex
@misc{cloudbench_2026,
	author       = { Sage Continuum and Francisco Lozano },
    affiliation  = { Northwestern University },
	title        = { CloudBench },
	year         = 2026,
	url          = { https://huggingface.co/datasets/sagecontinuum/CloudBench },
	doi          = { FILL_IN_DOI }, #TODO: Add DOI
	publisher    = { Hugging Face }
}
```

### Acknowledgments

We thank the creators and maintainers of:
- CCSN Dataset
- Sage Continuum

## Dataset Statistics

Please refer to the [EDA](summary/CloudBench_eda_analysis.ipynb) in the `summary/` directory.

## Hyperparameters in creating the dataset

Please refer to the [config_values.csv](summary/config_values.csv) file in the `summary/` directory for the values of the hyperparameters used in the dataset creation.

| value | description |
|-------|-------------|
query_plan_num_seeds | the number of seed images to use for query generation
query_plan_neg_total | the total number of negatives to generate for each query
query_plan_neg_hard | the number of hard negatives to generate for each query
query_plan_neg_nearmiss | the number of nearmiss negatives to generate for each query
query_plan_neg_easy | the number of easy negatives to generate for each query
query_plan_random_seed | the random seed used for reproducibility
controlled_tag_vocab | the controlled tag vocabulary for the CloudBench benchmark
vision_config.system_prompt | the system prompt for the vision annotation
vision_config.user_prompt | the user prompt for the vision annotation
vision_config.max_output_tokens | the maximum number of tokens for the vision annotation
vision_config.reasoning_effort | the reasoning effort for the vision annotation
vision_config.image_detail | the image detail level (low, medium, high)
vision_config.max_images_per_batch | the maximum number of images per vision batch shard
vision_config.max_concurrent_batches | the maximum number of batches to keep in flight
vision_config.completion_window | the completion window for the batch
vision_config.price_per_million_input_tokens | the price per million input tokens for the vision annotation
vision_config.price_per_million_output_tokens | the price per million output tokens for the vision annotation
vision_config.price_per_million_cached_input_tokens | the price per million cached input tokens for the batch
vision_config.price_per_million_image_input_tokens | the price per million image input tokens for the batch
vision_config.price_per_million_image_output_tokens | the price per million image output tokens for the batch
vision_config.vision_metadata_columns | the existing metadata columns to include in the vision annotation
judge_config.system_prompt | the system prompt for the judge
judge_config.user_prompt | the user prompt for the judge
judge_config.max_output_tokens | the maximum number of tokens for the judge
judge_config.reasoning_effort | the reasoning effort for the judge
judge_config.max_images_per_batch | the maximum number of images per judge batch shard
judge_config.max_concurrent_batches | the maximum number of batches to keep in flight
judge_config.completion_window | the completion window for the batch
judge_config.price_per_million_input_tokens | the price per million input tokens for the judge
judge_config.price_per_million_output_tokens | the price per million output tokens for the judge
judge_config.price_per_million_cached_input_tokens | the price per million cached input tokens for the judge
judge_config.price_per_million_image_input_tokens | the price per million image input tokens for the judge
judge_config.price_per_million_image_output_tokens | the price per million image output tokens for the judge
similarity_config.adapter | the adapter for the similarity scoring
similarity_config.model | the model for the similarity scoring
similarity_config.col_name | the column name for the similarity score
similarity_config.device | the device to run the similarity scoring on
similarity_config.use_safetensors | whether to use safetensors for the similarity scoring

## References

```
Liu, Pu. (2019). Cirrus Cumulus Stratus Nimbus (CCSN) Database (V2) [Data set]. Harvard Dataverse. https://doi.org/10.7910/DVN/CADDPD

Catlett, C. E., P. H. Beckman, R. Sankaran, and K. K. Galvin, 2017: Array of Things: A Scientific Research Instrument in the Public Way: Platform Design and Early Lessons Learned. Proceedings of the 2nd International Workshop on Science of Smart City Operations and Platforms Engineering, 26–33. https://doi.org/10.1109/ICSENS.2016.7808975
```