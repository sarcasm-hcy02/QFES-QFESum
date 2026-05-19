# RAT Retriever

This directory contains the implementation of the RAT retriever for Query-Focused Event Summarization (QFES). RAT stands for **Retrieval with Adaptive Thresholding**. It retrieves query-relevant documents from a thematic event corpus and provides the input document set for the downstream SHC summarizer.

The retriever first samples a subset of documents from the corpus and uses a local LLM to judge their relevance to the given query. It then computes semantic similarity scores between the query and documents using a BGE encoder and learns an adaptive similarity threshold based on the LLM-labeled samples. Finally, it applies the learned threshold to the full corpus and re-checks gray-zone documents with the LLM through sentence-level relevance judgment.

## 1. Environment

The code was tested with:

```text
Python 3.11.11
```

We recommend creating a separate Conda environment:

```bash
conda create -n rat-retriever python=3.11.11
conda activate rat-retriever
pip install -r requirements.txt
```

If NLTK sentence tokenization is used, please download the required resource before running the code:

```bash
python -c "import nltk; nltk.download('punkt')"
```

## 2. Dependencies

The main Python dependencies include:

```text
torch
transformers
safetensors
tqdm
requests
nltk
scikit-learn
numpy
```

The full dependency list is provided in:

```text
requirements.txt
```

## 3. Local LLM Service

RAT uses a local LLM for document relevance annotation and gray-zone document verification. By default, the implementation calls an Ollama service.

Please make sure that Ollama is installed, the required model is available, and the service is running:

```bash
ollama pull qwen2.5:7b
ollama serve
```

The default Ollama endpoint is:

```text
http://127.0.0.1:11434
```

The LLM model name and service endpoint can be modified in the corresponding LLM-related scripts if needed.

## 4. File Structure

The recommended file structure of this module is:

```text
RAT/
├── run_rat_retrieval.py              # Main entry script
├── BGE_calculate_id_list_score.py    # BGE-based similarity scoring
├── LLM_id_list_function.py           # LLM-based relevance annotation for sampled documents
├── LLM_retrieve_function.py          # LLM-based sentence-level judgment for gray-zone documents
├── sample_doc_ids.py                 # Random document sampling
├── requirements.txt                  # Python dependencies
└── README.md                         # Module documentation
```

Please make sure that the file names are consistent with the imports in `run_rat_retrieval.py`. For example:

```python
from BGE_calculate_id_list_score import calculate_similarity_scores
from LLM_id_list_function import check_relevance
from LLM_retrieve_function import llm_sentencewise_relevance
from sample_doc_ids import sample_document_ids
```

Therefore, the actual file names should not contain extra suffixes such as `(1)`, spaces, or other automatically generated marks.

## 5. Input Data Format

The input corpus should be a JSON file containing a list of documents. Each document should contain at least the following fields:

```json
[
  {
    "id": 1,
    "text": "Document text here."
  },
  {
    "id": 2,
    "text": "Another document text."
  }
]
```

The default document ID field is `id`, and the default document text field is `text`.

The helper function `_build_id2text()` in the main script provides limited compatibility for alternative field names such as `uid`, `doc_id`, `content`, or `body`. However, the BGE similarity scoring module expects the standard `id` and `text` fields by default. Therefore, the recommended input format is still:

```text
id + text
```

## 6. BGE Model Preparation

This module uses a local BGE model to compute query-document semantic similarity. Please download the model and place it in a local directory, for example:

```text
./model/BGE_large_v1.5
```

Then set the model path in `BGE_calculate_id_list_score.py`:

```python
tokenizer = AutoTokenizer.from_pretrained("./model/BGE_large_v1.5")
model = AutoModel.from_pretrained("./model/BGE_large_v1.5", use_safetensors=False)
```

It is recommended to use relative paths or properly expanded user paths. For example, avoid using:

```text
~./model/BGE_large_v1.5
```

If the model is placed under the home directory, use `~/model/BGE_large_v1.5` together with `os.path.expanduser()` or `Path(...).expanduser()`.

## 7. Usage

Before running the retriever, specify the input corpus path in `run_rat_retrieval.py`:

```python
json_file_path = "/path/to/your/corpus.json"
```

Then set the sampling ratio and random seed:

```python
SAMPLE_RATIO = 0.10
RANDOM_SEED = 42
```

The script automatically calls `sample_document_ids()` from `sample_doc_ids.py` to sample document IDs from the corpus. Therefore, there is no need to manually run the sampling script or manually copy the sampled ID list.

Next, configure retrieval tasks in `TASKS`:

```python
TASKS = [
    {
        "query": "Your query here",
        "output_json": "/path/to/output.json",
        "gold_path": "/path/to/gold.json"
    }
]
```

The fields are:

- `query`: the input query for retrieval;
- `output_json`: the path where the retrieved documents will be saved;
- `gold_path`: the reference relevant-document file, reserved for evaluation or analysis.

Then run:

```bash
python run_rat_retrieval.py
```

## 8. Retrieval Workflow

The overall retrieval workflow is as follows:

1. Randomly sample a subset of document IDs from the input corpus.
2. Use the local LLM to annotate the relevance of sampled documents with respect to the query.
3. Use BGE to compute semantic similarity scores between the query and sampled documents.
4. Learn an adaptive similarity threshold based on LLM labels and BGE scores.
5. Use BGE to compute similarity scores for the full corpus.
6. Directly keep documents with scores higher than `threshold + margin`.
7. Directly filter out documents with scores lower than `threshold - margin`.
8. Send documents within the gray zone `[threshold - margin, threshold + margin]` to the LLM for sentence-level relevance verification.
9. Save the final retrieved documents.

## 9. Output Format

The output file is a JSON file containing the retrieved documents. Each selected document keeps its original fields and is additionally assigned two fields:

```json
{
  "_similarity": 83.251,
  "_selected_by": "score"
}
```

The `_selected_by` field indicates how the document is selected:

```text
score
```

means the document is directly selected by the high-confidence BGE similarity score.

```text
llm
```

means the document is selected after LLM-based gray-zone verification.

## 10. Important Parameters

The main configurable parameters in `run_rat_retrieval.py` include:

```python
SAMPLE_RATIO = 0.10      # Ratio of sampled documents for LLM annotation
RANDOM_SEED = 42         # Random seed for document sampling
MARGIN = 3.0             # Width of the gray-zone boundary
```

The LLM progress display can also be configured:

```python
LLM_PROGRESS_MODE = "first_n"
LLM_PROGRESS_N = 5
LLM_PROGRESS_P = 0.10
LLM_PROGRESS_K = 5
```

## 11. Notes

This module is designed as an experimental retrieval component for QFES rather than a standalone Python package. It is mainly used for paper reproduction, QFESum experiments, and comparison with baseline methods.

For stable reproduction, please make sure that:

1. The input corpus follows the required JSON format.
2. The BGE model path is correctly configured.
3. The local Ollama service is running.
4. The LLM model name matches the model used in the code.
5. The output directory exists before writing results.
6. NLTK resources have been downloaded if sentence tokenization is used.

For future extension, the path settings, model names, sampling ratio, and margin can be further moved to command-line arguments or a configuration file.