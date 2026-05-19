# QFESum Dataset: Structure and Usage

## Overview

The **QFESum** dataset is designed for **Query-Focused Event Summarization (QFES)**.
It consists of multiple large-scale real-world event topics, each containing query-specific document collections, extracted events, and reference summaries.

The dataset includes the following **eight topics**:

* `bpoil` (BP Oil Spill)
* `finan` (Global Financial Crisis)
* `iraq` (Iraq War)
* `syria_t17`
* `egypt`
* `libya`
* `syria_crisis`
* `yemen`

All topics share the **same directory structure and data format**.

---

## Directory Structure

Each topic (e.g., `bpoil/`) contains **four components**:

```
bpoil/
├── bpoil.json
├── standard_documents/
├── QFS.json
└── timelines.json
```

---

## File Descriptions

### 1. `bpoil.json` — Full Document Collection

This file contains **all source documents** for the topic.

* Includes the complete document corpus before query-specific filtering
* Serves as the **raw input pool** for retrieval and summarization

---

### 2. `standard_documents/` — Query-Specific Documents and Events

This folder contains **query-level annotated data**.

* Each `.json` file corresponds to **one query**
* File names are **query abbreviations** (see `QFS.json` for mapping)

Each file includes:

* A set of **documents relevant to the query**
* **Event-level information** extracted from those documents

👉 This is the **core data** used for training and evaluating QFES systems.

---

### 3. `QFS.json` — Query Definitions and Reference Summaries

This file provides metadata for all queries in the topic:

* `query`: full natural language query
* `abbreviation`: short identifier used as filename
* `reference_summary`: gold summary for the query

👉 This file links:

* query text
* query abbreviation
* ground-truth summary

---

### 4. `timelines.json` — Original Timeline Summaries

This file contains **timeline-style summaries** from the original dataset.

* Topic-level chronological summaries
* Not query-specific
* Useful for comparison with timeline summarization methods

---

## Data Usage

### Step 1: Select a Topic

Choose one topic (e.g., `bpoil`).

---

### Step 2: Load Query Information

From `QFS.json`:

* Obtain the query text
* Map query → abbreviation
* Load the reference summary

---

### Step 3: Load Query-Specific Data

From `standard_documents/{abbr}.json`:

* Retrieve relevant documents
* Access extracted events

---

### Step 4: Generate Summary

Use:

* query
* query-specific documents/events

to generate a summary, and compare it with:

* `reference_summary` in `QFS.json`

---

## Notes

* All eight topics follow the **same format**, ensuring consistency across experiments
* Query abbreviations are **uniquely mapped** in `QFS.json`
* Event annotations are **query-aware**, not global

---

## Summary

QFESum provides:

* Multi-topic event corpora
* Query-specific document filtering
* Event-level annotations
* High-quality reference summaries

It is designed to support research in:

* Query-focused summarization
* Retrieval-augmented summarization
