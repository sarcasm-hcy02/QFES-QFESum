# QFES and QFESum

This repository contains two main components: the QFESum dataset and the QFES summarization method. QFESum is a dataset constructed for the Query-Focused Event Summarization (QFES) task. QFES is the proposed query-focused event summarization method, consisting of the RAT retriever and the SHC summarizer. This repository also includes test code and the experimental code for three baseline methods.

## 1. Repository Structure

```text
.
├── QFESum/
│   ├── README.md
│   ├── bpoil/
│   ├── finan/
│   ├── iraq/
│   ├── syria_t17/
│   ├── egypt/
│   ├── libya/
│   ├── syria_crisis/
│   └── yemen/
│
├── QFES/
│   ├── RAT/
│   ├── SHC/
│   ├── test/
│   ├── unstructured-evidence/
│   ├── graphrag-main/
│   └── LLM-TLS-main/
│
├── README.md
└── requirements.txt
```

## 2. QFESum Dataset

The `QFESum/` directory contains the QFESum dataset constructed for the Query-Focused Event Summarization task. Unlike general multi-document summarization, QFES requires a system to retrieve query-relevant documents from a thematic event corpus and generate a summary focused on the specific event aspect described by the given query.

Each thematic corpus usually contains the following files or directories:

```text
theme_name/
├── theme_name.json
├── QFS.json
├── timelines.json
└── standard_documents/
```

The main files are described as follows:

- `theme_name.json`: all documents in the thematic corpus.
- `QFS.json`: query abbreviations, full query descriptions, and query-focused reference summaries.
- `timelines.json`: the original timeline summaries.
- `standard_documents/`: query-specific relevant document sets and event annotations.

QFESum currently contains multiple thematic event corpora, including:

```text
bpoil
finan
iraq
syria_t17
egypt
libya
syria_crisis
yemen
```

For detailed field descriptions, data formats, and examples, please refer to:

```text
QFESum/README.md
```

## 3. QFES Method

The `QFES/` directory contains the proposed QFES method and related experimental code. The method mainly consists of two core modules:

```text
QFES/
├── RAT/
├── SHC/
├── test/
├── unstructured-evidence/
├── graphrag-main/
└── LLM-TLS-main/
```

### 3.1 RAT Retriever

`QFES/RAT/` contains the retriever module. RAT is used to retrieve query-relevant documents from a thematic event corpus and produce a query-specific relevant document set.

The main goals of this module are:

1. To compute the relevance between queries and documents.
2. To select relevant documents using adaptive thresholds.
3. To further examine gray-zone documents.
4. To provide query-focused input document sets for the SHC summarizer.

For detailed usage, please refer to:

```text
QFES/RAT/README.md
```

### 3.2 SHC Summarizer

`QFES/SHC/` contains the summarizer module. SHC generates query-focused event summaries based on event extraction, event representation, graph-based structural modeling, and hierarchical clustering.

The main goals of this module are:

1. To extract query-focused events from the retrieved documents.
2. To construct an event graph and model the event structure.
3. To cluster, rank, and select events.
4. To generate the final query-focused event summary.

For detailed usage, please refer to:

```text
QFES/SHC/README.md
```

### 3.3 Test Code

`QFES/test/` contains test code for checking the functional correctness of different QFES modules and the completeness of the experimental workflow.

This part is mainly used to:

1. Test the RAT retrieval module.
2. Test the SHC summarization module.
3. Check input and output formats.
4. Verify whether the experimental pipeline can run correctly.

For detailed usage, please refer to:

```text
QFES/test/README.md
```

## 4. Baseline Methods

This repository also includes the experimental code for three baseline methods:

```text
QFES/unstructured-evidence/
QFES/graphrag-main/
QFES/LLM-TLS-main/
```

### 4.1 Unstructured Evidence

`QFES/unstructured-evidence/` contains the baseline experimental code based on the evidence extraction and LLM summarization paradigm.

This method first extracts query-relevant evidence from input documents and then generates the final summary based on the extracted evidence.

For detailed usage, please refer to:

```text
QFES/unstructured-evidence/README.md
```

### 4.2 GraphRAG

`QFES/graphrag-main/` contains the experimental code for the GraphRAG baseline.

This method uses a graph-enhanced retrieval and generation framework and is included as a baseline for comparison with the proposed QFES method.

For detailed usage, please refer to:

```text
QFES/graphrag-main/README.md
```

### 4.3 LLM-TLS

`QFES/LLM-TLS-main/` contains the experimental code for the LLM-TLS baseline.

This method is mainly used for timeline or event summarization based on event similarity, LLM-based semantic judgment, and clustering strategies.

For detailed usage, please refer to:

```text
QFES/LLM-TLS-main/README.md
```

## 5. Environment

The code in this repository was tested with:

```text
Python 3.11.11
```

We recommend creating an isolated Conda environment:

```bash
conda create -n qfes python=3.11.11
conda activate qfes
```

Then install the overall dependencies:

```bash
pip install -r requirements.txt
```

If a specific submodule contains its own `requirements.txt`, the dependencies can also be installed separately:

```bash
cd QFES/RAT
pip install -r requirements.txt
```

or:

```bash
cd QFES/SHC
pip install -r requirements.txt
```

## 6. Basic Usage

We recommend running the experiments in the following order.

### Step 1: Prepare the Dataset

First, make sure that the `QFESum/` dataset is placed in the repository and that each thematic corpus contains the required files:

```text
theme_name.json
QFS.json
timelines.json
standard_documents/
```

### Step 2: Run the RAT Retriever

Enter the RAT module:

```bash
cd QFES/RAT
```

Run the retrieval scripts according to the module-level README. The output of RAT is usually used as the input to SHC.

### Step 3: Run the SHC Summarizer

Enter the SHC module:

```bash
cd QFES/SHC
```

Run the summarization scripts according to the module-level README to generate query-focused event summaries.

### Step 4: Run Evaluation or Tests

Enter the test module:

```bash
cd QFES/test
```

Run the test scripts according to the module-level README to check the retrieval, summarization, and evaluation workflow.

## 7. Notes for Anonymous Review

This repository is prepared for anonymous review. Author-identifying information has been removed from the code, documentation, and metadata.

When using this repository for review or reproduction, please refer to the module-level README files for detailed instructions.

## 8. License and Data Usage

Please refer to the final camera-ready version or the official public release for complete license and data usage information.