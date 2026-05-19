# Evaluation Module for Query-Focused Event Summarization

This directory contains evaluation scripts for Query-Focused Event Summarization (QFES). It supports two types of evaluation:

1. **Automatic text-level evaluation**, including ROUGE, BLEU, METEOR, and BERTScore.
2. **LLM-based event-level evaluation**, which evaluates whether each generated event sentence can be matched to and fully supported by a reference event sentence.

The expected input is a JSON file containing generated summaries and reference summaries.

## 1. Environment

The code was tested with:

```text
Python 3.11.11
```

We recommend creating an isolated Conda environment:

```bash
conda create -n qfes-eval python=3.11.11
conda activate qfes-eval
pip install -r requirements.txt
```

## 2. Dependencies

The main Python dependencies include:

```text
numpy
requests
tqdm
nltk
rouge-score
sacrebleu
bert-score
transformers
torch
openai
```

The full dependency list is provided in:

```text
requirements.txt
```

## 3. File Structure

The recommended file structure of this module is:

```text
test/
├── ROUGE_BLUE_METEOR_BERTScore.py   # ROUGE, BLEU, METEOR, and BERTScore evaluation
├── LLM_api_score_batch.py           # LLM-based event-level Precision/Recall/F1 evaluation
├── requirements.txt                 # Python dependencies
└── README.md                        # Module documentation
```

If a local transformer model is used for BERTScore, the model directory can be placed outside this module and specified in the script. Large local model files should not be committed to the GitHub repository.

## 4. Input Format

Both evaluation scripts expect a JSON file containing a list of summary pairs. Each item should contain:

```json
[
  {
    "generated_summary": "Generated query-focused event summary.",
    "reference_summary": "Reference summary."
  }
]
```

where:

- `generated_summary` is the summary produced by the system.
- `reference_summary` is the human-written or gold reference summary.

Records without valid `generated_summary` or `reference_summary` fields will be skipped.

## 5. Automatic Metric Evaluation

The script `ROUGE_BLUE_METEOR_BERTScore.py` computes the following automatic text-level metrics:

- ROUGE-1
- ROUGE-2
- ROUGE-L
- ROUGE-4
- BLEU
- METEOR
- BERTScore

### 5.1 Configure the Input File

Modify the following variable in `ROUGE_BLUE_METEOR_BERTScore.py`:

```python
JSON_PATH = "./QFESum/libya/EG-QFS/summary.json"
```

The input file should follow the format described in Section 4.

### 5.2 Configure the BERTScore Model

By default, BERTScore can use a local `roberta-large` model:

```python
local_model_path = "./model/roberta-large"
```

Make sure this directory contains the required model files, such as `config.json`, tokenizer files, and model weights.

If you want to use a Hugging Face model name instead of a local model path, modify the corresponding `model_type` argument in the BERTScore call.

Large local model files should be excluded from Git tracking, for example by adding the following entry to the root `.gitignore`:

```text
model/
```

### 5.3 Run

```bash
python ROUGE_BLUE_METEOR_BERTScore.py
```

The script prints average ROUGE, BLEU, METEOR, and BERTScore results.

## 6. LLM-Based Event-Level Evaluation

The script `LLM_api_score_batch.py` computes event-level Precision, Recall, and F1 by matching generated event sentences against reference event sentences.

For each generated event sentence, the script checks whether one reference event sentence fully supports it. Each reference sentence can be matched at most once. The final scores are computed as:

```text
Precision = matched generated sentences / generated sentences
Recall    = matched reference sentences / reference sentences
F1        = 2 * Precision * Recall / (Precision + Recall)
```

This metric is designed to evaluate event-level semantic alignment rather than surface token overlap.

### 6.1 Configure the Input Files

Modify the following list in `LLM_api_score_batch.py`:

```python
INPUT_FILE_PATHS = [
    "./QFESum/libya/EG-QFS/summary.json"
]
```

Multiple files can be evaluated together:

```python
INPUT_FILE_PATHS = [
    "./QFESum/libya/EG-QFS/summary.json",
    "./QFESum/yemen/EG-QFS/summary.json"
]
```

### 6.2 Configure the LLM API

The script uses an OpenAI-compatible API client. For example, when using the DashScope-compatible endpoint, the client can be configured as:

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
```

The default model can be set in the script, for example:

```python
model = "deepseek-v3"
```

Set the API key through an environment variable before running the script:

```bash
export DASHSCOPE_API_KEY="your_api_key_here"
```

Do not hard-code API keys in the source code.

### 6.3 Run

```bash
python LLM_api_score_batch.py
```

The script prints file-level and average event-level Precision, Recall, and F1 scores.

## 7. Sentence Splitting

`LLM_api_score_batch.py` uses a simple period-based sentence splitter for event-level matching. Very short sentences are filtered out by a minimum word threshold.

If summaries contain abbreviations, non-standard punctuation, or complex sentence boundaries, replacing the default splitter with a more robust sentence segmentation method is recommended.

`ROUGE_BLUE_METEOR_BERTScore.py` uses simple English tokenization for METEOR to avoid dependency on NLTK Punkt.

## 8. Overall Pipeline

The automatic metric evaluation pipeline is:

```text
Generated summary + Reference summary
        ↓
ROUGE_BLUE_METEOR_BERTScore.py
        ↓
ROUGE / BLEU / METEOR / BERTScore
```

The LLM-based event-level evaluation pipeline is:

```text
Generated summary + Reference summary
        ↓
LLM_api_score_batch.py
        ↓
Event-level Precision / Recall / F1
```

## 9. Notes

This module is designed for evaluating QFES outputs and comparing different summarization methods under the same input format.

For stable reproduction, please make sure that:

1. The input JSON files follow the required format.
2. Local model paths are correctly configured if local models are used.
3. API keys are provided through environment variables.
4. Large local model files are not committed to the GitHub repository.
5. The same sentence-splitting strategy is used when comparing different systems with LLM-based event-level metrics.