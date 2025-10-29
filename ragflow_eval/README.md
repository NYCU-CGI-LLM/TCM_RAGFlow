# RAGAS Evaluation for RAGFlow

This module provides RAGAS-style evaluation capabilities for the RAGFlow retrieval system using the RAGFlow SDK.

## Features

- **SDK-based evaluation**: Uses RAGFlow Python SDK
- **Config-driven**: YAML configuration with proper comment support
- **Comprehensive metrics**: Recall@K, Precision@K
- **CLI integration**: Run evaluations via command line
- **Batch processing**: Evaluate entire datasets with progress tracking

## Installation
```
uv venv --python 3.11
uv sync
source .venv/bin/activate
```


## Quick Start

### 1. Prepare Your Dataset

Create a JSON file with your evaluation data:

```json
[
  {
    "prompt": "患者临床信息：主诉：反复胸闷憋喘2年余...",
    "expected_doc_id": 123
  },
  {
    "prompt": "Another query here",
    "expected_doc_id": 456
  }
]
```

**With metadata** (like TCM dataset):
```json
[
  {
    "user_id": "123",
    "prompt": "...",
    "expected_doc_id": 593,
    "expected_answer": "...",
    "format_type": "..."
  }
]
```

### 2. Create Configuration File

Copy `config/example_config.yaml` and modify:

> Choose exactly one evaluation mode per config: `retrieval` **or** `generation`.

```yaml
# Example configuration for RAGFlow evaluation
retrieval:
  api_key: "ragflow-{your_api_key}"
  dataset_id: "{your_dataset_id}"
  # OR use dataset_name instead:
  # dataset_name: "My Knowledge Base"
  
  size: 5  # Number of chunks to retrieve
  # Optional: specify which metrics to calculate
  metrics:
    - recall
    - precision
    - f1
  # Optional: specify K values for @K metrics
  k_values:
    - 1
    - 3
    - 5
  
  # Optional: only add if your server runs on a different URL
  # base_url: "http://your-server:8080"

dataset:
  path: "./dataset/your_dataset.json"
  # limit: 100

output:
  results_path: "results/evaluation_results.json"
```

```yaml
# Example configuration for generation-focused evaluation
generation:
  api_key: "ragflow-{your_api_key}"
  chat_name: "My Assistant"  # or chat_id: "{your_chat_id}"
  # size: 8  # Optional: override retrieval chunk count for this evaluation run
  metrics:
    - recall
    - precision
  k_values:
    - 1
    - 3
    - 5

dataset:
  path: "./dataset/your_dataset.json"

output:
  results_path: "results/generation_results.json"
```

### 3. Run Evaluation

#### Via Standalone Script (Recommended):

```bash
python evaluation/run_evaluation.py --config evaluation/config/retrieval_example.yaml
```

#### Via Python:

```python
from evaluation.ragas_evaluation import run_evaluation

results = run_evaluation('path/to/config.yaml')
```

## Configuration Options

### Retrieval Settings

- `api_key`: Your RAGFlow API key **(required)**
- `dataset_id`: Knowledge base/dataset UUID to retrieve from **(required if `dataset_name` not provided)**
- `dataset_name`: Knowledge base/dataset friendly name to retrieve from **(required if `dataset_id` not provided)**
  - **Tip:** Use `dataset_name` for better readability. The system will automatically resolve it to the UUID.
- `base_url`: RAGFlow server base URL (default: `http://localhost:9380`) **(optional)**
- `size`: Number of chunks to retrieve (default: 5)
- `vector_similarity_weight`: Weight for vector similarity (0-1, default: 1.0) **(optional)**
- `dynamic_rerank_limit`: Whether to use dynamic rerank limit (true/false, default: true) **(optional)**

### Dataset Settings

- `path`: Path to dataset JSON file
- `limit`: Maximum number of samples to evaluate (null = all)

### Output Settings

- `results_path`: Where to save evaluation results

## Metrics Explained

### Recall@K
Proportion of relevant documents that are retrieved in top K results.
- **Formula**: (Relevant Retrieved) / (Total Relevant)
- **Range**: 0.0 to 1.0 (higher is better)

### Precision@K
Proportion of retrieved documents that are relevant in top K results.
- **Formula**: (Relevant Retrieved) / (Total Retrieved)
- **Range**: 0.0 to 1.0 (higher is better)

<!-- ### F1@K -->
<!-- Harmonic mean of Precision@K and Recall@K. -->
<!-- - **Formula**: 2 × (Precision × Recall) / (Precision + Recall) -->
<!-- - **Range**: 0.0 to 1.0 (higher is better) -->
<!---->
<!-- ### NDCG@K -->
<!-- Normalized Discounted Cumulative Gain - considers ranking order. -->
<!-- - **Range**: 0.0 to 1.0 (higher is better) -->
<!---->
<!-- ### MRR (Mean Reciprocal Rank) -->
<!-- Reciprocal of the rank of the first relevant document. -->
<!-- - **Range**: 0.0 to 1.0 (higher is better) -->
<!---->
### Hit Rate
Whether at least one relevant document was retrieved.
- **Range**: 0.0 or 1.0

## Output Format

Results are saved as JSON with:

```json
{
  "config": {
    "dataset_path": "...",
    "dataset_size": 100,
    "top_k": 5,
    ...
  },
  "aggregate_metrics": {
    "recall@5": {
      "mean": 0.85,
      "median": 0.90,
      "std": 0.12,
      "min": 0.20,
      "max": 1.0
    },
    ...
  },
  "detailed_results": [
    {
      "query": "...",
      "ground_truth_ids": [385, 540],
      "retrieved_ids": [385, 123, 540],
      "retrieval_time": 0.234,
      "metrics": {
        "recall@5": 1.0,
        "precision@5": 0.666,
        ...
      }
    }
  ],
  "timestamp": "2025-10-08 12:34:56"
}
```

## Examples


### Limited Sample Evaluation (Testing)

Modify config to include `limit`:
```json
{
  "dataset": {
    "path": "./dataset/large_dataset.json",
    "limit": 10
  }
}
```
### Basic Evaluation

```bash
python run_evaluation.py --config config/tcm/retrieval_qwen3_4b_v2.yaml
```

### Example Output:

```
================================================================================
RAGAS Evaluation Summary
================================================================================

Dataset: ./dataset/tcm_sd_test_rc_direct.json
Samples evaluated: 100
Top K: 5

--------------------------------------------------------------------------------
Aggregate Metrics
--------------------------------------------------------------------------------

recall@5:
  Mean:   0.8500
  Median: 0.9000
  Std:    0.1200
  Min:    0.2000
  Max:    1.0000

precision@5:
  Mean:   0.7800
  Median: 0.8000
  Std:    0.1500
  Min:    0.2000
  Max:    1.0000
```

## Troubleshooting

### "Cannot extract dataset_id from API URL"
Ensure your `api_url` follows the format:
```
http://localhost:9380/api/v1/retrieval_simple_rag/{dataset_id}
```

### "Dataset file not found"
Check that the `path` in config is relative to where you run the command, or use absolute paths.

### "API key authentication failed"
Verify your API key is correct and has proper permissions.


## Development

### Using RAGAS Metrics

The evaluation system now uses **RAGAS framework metrics** for both retrieval and generation evaluation.

**Retrieval Metrics (ID-based, FREE):**
- `context_precision@K` - What fraction of retrieved docs are relevant?
- `context_recall@K` - What fraction of relevant docs were retrieved?

**Generation Metrics:**
- `ExactMatch` - Case-insensitive string comparison with ground truth

See `RAGAS_MIGRATION.md` for full details and migration guide.

## License

Copyright 2024 The InfiniFlow Authors. Licensed under Apache 2.0.
