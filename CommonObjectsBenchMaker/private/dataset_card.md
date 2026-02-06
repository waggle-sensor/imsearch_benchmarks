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
    dtype:
      class_label:
        names:
          - non_relevant
          - relevant
  - name: doi
    dtype: string
  - name: license
    dtype: string
  - name: summary
    dtype: string
  - name: tags
    sequence: string
  - name: confidence
    struct:
    - name: animal_present
      dtype: float64
    - name: artificial_lighting
      dtype: float64
    - name: environment_type
      dtype: float64
    - name: food_present
      dtype: float64
    - name: lighting
      dtype: float64
    - name: multiple_objects
      dtype: float64
    - name: occlusion_present
      dtype: float64
    - name: outdoor_scene
      dtype: float64
    - name: person_present
      dtype: float64
    - name: rural_scene
      dtype: float64
    - name: text_visible
      dtype: float64
    - name: urban_scene
      dtype: float64
    - name: vehicle_present
      dtype: float64
    - name: viewpoint
      dtype: float64
  - name: urban_scene
    dtype: bool
  - name: rural_scene
    dtype: bool
  - name: outdoor_scene
    dtype: bool
  - name: vehicle_present
    dtype: bool
  - name: person_present
    dtype: bool
  - name: animal_present
    dtype: bool
  - name: food_present
    dtype: bool
  - name: text_visible
    dtype: bool
  - name: multiple_objects
    dtype: bool
  - name: artificial_lighting
    dtype: bool
  - name: occlusion_present
    dtype: bool
  - name: viewpoint
    dtype:
      class_label:
        names:
          - eye_level
          - overhead
          - close_up
          - distant
          - street_view
          - top_down
          - oblique
          - side_view
          - first_person
          - skyward
          - other
          - unknown
  - name: lighting
    dtype:
      class_label:
        names:
          - day
          - night
          - dusk
          - indoor
          - shadow
          - bright
          - backlit
          - mixed
          - other
          - unknown
  - name: environment_type
    dtype:
      class_label:
        names:
          - indoor
          - outdoor
          - urban
          - suburban
          - rural
          - residential
          - commercial
          - industrial
          - recreational
          - natural
          - park
          - beach
          - other
          - unknown
  - name: clip_score
    dtype: float64
  - name: image
    dtype: image
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
task_categories:
- image-classification
language:
- en
tags:
- image-retrieval
- benchmark
- computer-vision
- common-objects
- coco
size_categories:
- 10K<n<100K
pretty_name: Common Objects Benchmark
---

# CommonObjectsBench-private: A Benchmark Dataset for General Object Image Retrieval

## Dataset Description

CommonObjectsBench is a benchmark dataset for evaluating image retrieval systems on general objects and common scenes. The dataset consists of natural language queries paired with images, along with binary relevance labels indicating whether each image is relevant to the query. The dataset is designed to test retrieval systems' ability to find relevant images based on queries describing common objects and scenes.
>NOTE: This is the private version of the dataset, so urban images from sage nodes are included. The public version is available at [CommonObjectsBench on Hugging Face](https://huggingface.co/datasets/sagecontinuum/CommonObjectsBench). The public version does not include urban images. Please be careful when using the private dataset as the urban images are not allowed to be public.

![Image Sample](summary/random_image_sample.png)

### Dataset Summary

CommonObjectsBench contains:
- **Queries**: Natural language queries describing common objects and scenes
- **Images**: Real-world imagery from multiple sources (COCO 2017 dataset, Sage Continuum)
- **Relevance Labels**: Binary labels (0 = not relevant, 1 = relevant) for each query-image pair
- **Rich Metadata**: Comprehensive annotations including object descriptions, scene characteristics, and more
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
    "query_id": "commonobjectsbench_q001",
    "query_text": "A person riding a bicycle on a sunny day",
    "image_id": "coco/train2017/000000000139.jpg",
    "relevance_label": 1,
    "image": <PIL.Image.Image>,  # The actual image
    "license": "CC BY 4.0",
    "doi": "UNKNOWN",
    "tags": ["person", "bicycle", "outdoor", "day", "sunny", ...],
    "viewpoint": "eye_level",
    "lighting": "day",
    "environment_type": "urban",
    "multiple_objects": false,
    "artificial_lighting": false,
    "occlusion_present": false,
    "text_visible": false,
    "person_present": true,
    "animal_present": false,
    "food_present": false,
    "urban_scene": true,
    "rural_scene": false,
    "outdoor_scene": true,
    "vehicle_present": false,
    "confidence": {
        "viewpoint": 0.9,
        "lighting": 0.9,
        "environment_type": 0.9
        ...
    },
    "summary": "A person riding a bicycle on a sunny day in an urban setting.",
    "clip_score": 5.337447166442871
}
```

### Data Fields

- **query_id** (string): Unique identifier for the query (e.g., "commonobjectsbench_q001")
- **query_text** (string): Natural language query describing the target objects or scene
- **image_id** (string): Unique identifier for the image (relative path from source)
- **relevance_label** (int): Binary relevance label (0 = not relevant, 1 = relevant)
- **image** (Image): The actual image file
- **license** (string): License information for the image (e.g., "CC BY 4.0")
- **doi** (string): Digital Object Identifier for the source dataset
- **viewpoint** (string): Camera viewpoint (e.g., "eye_level", "overhead", "close_up", "distant", "street_view", "top_down", "oblique", "side_view", "first_person", "skyward", "other", "unknown")
- **lighting** (string): Lighting conditions (e.g., "day", "night", "dusk", "indoor", "shadow", "bright", "backlit", "mixed", "other", "unknown")
- **environment_type** (string): Type of environment (e.g., "indoor", "outdoor", "urban", "suburban", "rural", "residential", "commercial", "industrial", "recreational", "natural", "park", "beach", "other", "unknown")
- **multiple_objects** (bool): Whether more than one distinct object category is present
- **artificial_lighting** (bool): Whether the main lighting is artificial
- **occlusion_present** (bool): Whether the main subject is partially occluded
- **text_visible** (bool): Whether readable text is present in the image
- **person_present** (bool): Whether a person is present in the image
- **animal_present** (bool): Whether an animal is present in the image
- **food_present** (bool): Whether food is present in the image
- **urban_scene** (bool): Whether the image is an urban scene
- **rural_scene** (bool): Whether the image is a rural scene
- **outdoor_scene** (bool): Whether the image is an outdoor scene
- **vehicle_present** (bool): Whether a vehicle is present in the image
- **tags** (list of strings): Controlled vocabulary tags describing the image (12-18 tags per image)
- **confidence** (dict): Confidence scores (0-1) for annotations
- **summary** (string): Brief factual summary of the image (≤30 words)
- **clip_score** (float): Pre-computed CLIP similarity score between query and image

### Data Splits

The dataset is provided as a single split. Users can create their own train/validation/test splits as needed.

## Dataset Creation

### Curation Rationale

CommonObjectsBench was created to address the need for a standardized benchmark for evaluating image retrieval systems on general objects and common scenes. The dataset focuses on:

1. **Real-world diversity**: Images from multiple sources covering various objects, scenes, and environments
2. **General-purpose queries**: Queries describing common objects and everyday scenes
3. **Comprehensive annotations**: Rich metadata to support both retrieval evaluation and analysis

### Source Data

The dataset combines images from two sources:

1. **COCO 2017 Dataset** ([COCO](https://cocodataset.org/#home))
   - Large-scale object detection, segmentation, and captioning dataset
   - 80 object categories
   - DOI: 10.48550/arXiv.1405.0312

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

The dataset contains private imagery taken from urban sage nodes. Which means it may contain sensitive information such as personal information, private property, or other sensitive information. Please be careful when using the private dataset as the urban images are not allowed to be public.

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

The dataset was created using imsearch_benchmaker, a pipeline for creating image retrieval benchmarks. See [imsearch_benchmarks/CommonObjectsBenchMaker](https://github.com/waggle-sensor/imsearch_benchmarks/tree/main/CommonObjectsBenchMaker) for the code implementation.

### Licensing Information

Images in the dataset are licensed according to their source:
- **COCO Dataset**: See source for licensing information
- **Sage Continuum**: See source for licensing information

The dataset card and annotations are provided under CC BY 4.0.

### Acknowledgments

We thank the creators and maintainers of:
- COCO Dataset
- Sage Continuum

## Dataset Statistics

Please refer to the [EDA](summary/CommonObjectBench_eda_analysis.ipynb) in the `summary/` directory.

## Hyperparameters in creating the dataset

Please refer to the [config_values.csv](summary/config_values.csv) file in the [summary/](summary/) directory for the values of the hyperparameters used in the dataset creation.

| value | description |
|-------|-------------|
query_plan_num_seeds | the number of seed images to use for query generation
query_plan_neg_total | the total number of negatives to generate for each query
query_plan_neg_hard | the number of hard negatives to generate for each query
query_plan_neg_nearmiss | the number of nearmiss negatives to generate for each query
query_plan_neg_easy | the number of easy negatives to generate for each query
query_plan_random_seed | the random seed used for reproducibility
controlled_tag_vocab | the controlled tag vocabulary for the CommonObjectsBench benchmark
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
Lin, T.-Y., Maire, M., Belongie, S., Bourdev, L., Girshick, R., Hays, J., Perona, P., Ramanan, D., Zitnick, C. L., & Dollár, P. (2015). Microsoft COCO: Common Objects in Context. arXiv preprint arXiv:1405.0312. https://arxiv.org/abs/1405.0312

Catlett, C. E., P. H. Beckman, R. Sankaran, and K. K. Galvin, 2017: Array of Things: A Scientific Research Instrument in the Public Way: Platform Design and Early Lessons Learned. Proceedings of the 2nd International Workshop on Science of Smart City Operations and Platforms Engineering, 26–33. https://doi.org/10.1109/ICSENS.2016.7808975
```