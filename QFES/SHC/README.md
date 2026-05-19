# SHC Query-Focused Event Summarization

This directory contains the implementation of the SHC summarization module for Query-Focused Event Summarization (QFES). SHC generates query-focused event summaries from documents retrieved by RAT or other query-focused retrieval methods.

The pipeline consists of two main stages. First, a local LLM extracts query-relevant events from the retrieved documents. Second, the SHC module performs event embedding, clustering, representative event selection, and semantic deduplication to generate the final query-focused event summary.

## 1. Environment

The code was tested with:

```text
Python 3.11.11
```

We recommend creating an isolated Conda environment:

```bash
conda create -n shc-summary python=3.11.11
conda activate shc-summary
pip install -r requirements.txt
```

## 2. Dependencies

The main Python dependencies include:

```text
numpy
torch
transformers
sentence-transformers
scikit-learn
umap-learn
hdbscan
bertopic
rouge-score
requests
tqdm
nltk
pandas
```

The full dependency list is provided in:

```text
requirements.txt
```

## 3. External Resources

This module requires local embedding models, for example:

```text
./model/BGE_large_v1.5
./model/GTE_large
```

The two models are used for different purposes:

- `BGE_large_v1.5` is used for event embedding, event clustering, and query-event/topic similarity computation.
- `GTE_large` is used for semantic deduplication of generated summary sentences.

If different SentenceTransformer models are used, please modify the corresponding model paths in the scripts.

The event extraction stage also requires a local Ollama service. By default, the script uses:

```bash
ollama pull qwen2.5:7b
ollama serve
```

The default model name in `LLM_extract_batch.py` is:

```python
MODEL = "qwen2.5:7b"
```

To use another local LLM, modify the `MODEL` variable accordingly.

Local model files should not be committed to the GitHub repository. It is recommended to keep model directories ignored by `.gitignore`, for example:

```text
model/
```

## 4. File Structure

The recommended file structure of this module is:

```text
SHC/
├── bertopic_HDBSCAN_class.py      # Core SHC clustering and summarization module
├── LLM_extract_batch.py           # LLM-based query-focused event extraction
├── prompt_event_extract.py        # Prompt template for event extraction
├── sentence_simi.py               # Sentence-level semantic deduplication
├── run_summary.py                 # Entry script for SHC summarization and evaluation
├── requirements.txt               # Python dependencies
└── README.md                      # Module documentation
```

The local model directory and dataset directory can be placed outside this module or configured through paths in the scripts.

## 5. Input Format

This module contains two stages, and the expected input format is different for each stage.

### 5.1 Input Format for Event Extraction

`LLM_extract_batch.py` expects a JSON file containing retrieved documents. Each item should contain at least an `id` field and a `text` field:

```json
[
  {
    "id": 1,
    "text": "Document text ..."
  },
  {
    "id": 2,
    "text": "Another document text ..."
  }
]
```

where:

- `id` is the document ID.
- `text` is the document content.

The script reads the `text` field of each document and extracts query-relevant events according to the given query.

### 5.2 Input Format for SHC Summarization

`run_summary.py` and `EventClusterSummarizer` expect documents with an `events` field:

```json
[
  {
    "id": 1,
    "text": "Document text ...",
    "events": [
      {
        "eid": 1,
        "text": "Extracted event sentence."
      },
      {
        "eid": 2,
        "text": "Another extracted event sentence."
      }
    ]
  }
]
```

where:

- `eid` is the event ID.
- `text` is the extracted event text.

In the current version, `LLM_extract_batch.py` first stores LLM-extracted events as raw strings. After all documents in one task have been processed, the script automatically assigns globally incremental integer `eid` values to all extracted events and converts them into the following format:

```json
{
  "eid": 1,
  "text": "Extracted event sentence."
}
```

Therefore, the output of `LLM_extract_batch.py` can be directly used as the input of `run_summary.py`.

## 6. Running Event Extraction

First, define the extraction tasks in `LLM_extract_batch.py`:

```python
TASKS = [
    {
        "query": "Your query here",
        "input_path": "./QFESum/egypt/RAT/input.json"
    }
]
```

By default, the script writes the extracted events back to the same file specified by `input_path`.

If you do not want to overwrite the input file, add an optional `output_path` field:

```python
TASKS = [
    {
        "query": "Your query here",
        "input_path": "./QFESum/egypt/RAT/input.json",
        "output_path": "./QFESum/egypt/events/output.json"
    }
]
```

Then run:

```bash
python LLM_extract_batch.py
```

The script calls a local Ollama model to perform query-focused event extraction for each document. During extraction, the LLM only outputs event texts. After all documents in one task have been processed, the script automatically assigns globally incremental `eid` values to the extracted events.

The event extraction script also supports a blue-green switching strategy between two Ollama ports to reduce possible performance degradation during long-running inference. The default ports are:

```python
PRIMARY_PORT = 11434
SECONDARY_PORT = 11435
```

If only one Ollama service is used, please make sure that the port configuration in the script matches the actual local service.

## 7. Running SHC Summarization

After event extraction, run `run_summary.py` to generate SHC-based summaries.

First, define summarization tasks in `run_summary.py`:

```python
tasks = [
    {
        "json_path": "./QFESum/libya/RAT/European_Union.json",
        "query": "The response and interference of the European Union",
        "reference": "Reference summary here."
    }
]
```

where:

- `json_path` is the input JSON file containing extracted events.
- `query` is the current query.
- `reference` is the reference summary used for ROUGE evaluation.

If only summary generation is needed, the `reference` field and the reference-related evaluation code can be omitted or ignored.

Then specify the embedding model path:

```python
model_path = "./model/BGE_large_v1.5"
```

Run:

```bash
python run_summary.py
```

The program performs the following steps:

1. Load extracted events from the input JSON files.
2. Encode events using SentenceTransformer.
3. Reduce event embedding dimensions with UMAP.
4. Cluster events using HDBSCAN and BERTopic.
5. Compute the similarity between each topic and the query.
6. Split topics into high-, middle-, and low-relevance regions according to topic-query similarity.
7. Select representative center events from different relevance regions.
8. Deduplicate generated summary sentences using semantic similarity.
9. Output the final query-focused event summary.
10. Compute ROUGE scores if reference summaries are provided.

## 8. Main Parameters

The main parameters of `EventClusterSummarizer` include:

```python
threshold_high_ratio
threshold_mid_ratio
min_dist
min_cluster_size
min_samples
```

Their meanings are:

| Parameter | Description |
|---|---|
| `threshold_high_ratio` | Ratio of topics assigned to the high-relevance region |
| `threshold_mid_ratio` | Cumulative ratio of topics assigned to the high- and middle-relevance regions |
| `min_dist` | Distance-control parameter in UMAP |
| `min_cluster_size` | Minimum cluster size in HDBSCAN |
| `min_samples` | Density-control parameter in HDBSCAN |

Example:

```python
summarizer = EventClusterSummarizer(
    json_file_path=task["json_path"],
    query=task["query"],
    model_path="./model/BGE_large_v1.5",
    threshold_high_ratio=0.3,
    threshold_mid_ratio=0.5,
    min_dist=0.05,
    min_cluster_size=20,
    min_samples=20
)
```

## 9. Output Format

`run_summary.py` outputs a JSON file containing the generated summaries and, when available, reference summaries:

```json
[
  {
    "generated_summary": "Generated query-focused event summary.",
    "reference_summary": "Reference summary."
  }
]
```

If only generated summaries are needed, the output can be simplified as:

```json
[
  {
    "query": "Your query here",
    "generated_summary": "Generated query-focused event summary."
  }
]
```

## 10. Overall Pipeline

The overall SHC pipeline is:

```text
Retrieved documents
        ↓
LLM_extract_batch.py
        ↓
Query-relevant extracted events with eid
        ↓
run_summary.py
        ↓
SHC event clustering and center-event selection
        ↓
Semantic deduplication
        ↓
Query-focused event summary
```

## 11. Notes

This module is designed as an experimental summarization component for QFES rather than a standalone Python package. It is mainly used for paper reproduction, QFESum experiments, and comparison with baseline methods.

For stable reproduction, please make sure that:

1. The input documents have already been filtered by RAT or another query-focused retrieval method.
2. The local Ollama service is running before event extraction.
3. The local LLM model name matches the model name used in `LLM_extract_batch.py`.
4. The local SentenceTransformer model paths are correctly configured.
5. The input JSON files follow the required format.
6. The output directories exist before writing results.
7. Large local model files are not committed to the GitHub repository.

For future extension, the path settings, model names, task configurations, and hyperparameters can be further moved to command-line arguments or a configuration file.