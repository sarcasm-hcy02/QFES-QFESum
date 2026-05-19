# QFES 与 QFESum

本仓库包含两个主要部分：QFESum 数据集和 QFES 摘要算法。QFESum 是面向 Query-Focused Event Summarization（QFES）任务构建的数据集；QFES 是本文提出的查询聚焦事件摘要方法，包括 RAT 检索器和 SHC 摘要器。此外，本仓库还包含测试代码以及三个 baseline 方法的实验代码。

## 1. 仓库结构

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

## 2. QFESum 数据集

`QFESum/` 文件夹包含本文构建的 QFESum 数据集。该数据集用于 Query-Focused Event Summarization 任务。与普通多文档摘要任务不同，QFES 要求系统在一个主题事件语料库中，根据给定 query 检索与该 query 相关的文档，并生成聚焦于该 query 所指定事件方面的摘要。

每个主题语料库通常包含以下文件或子目录：

```text
theme_name/
├── theme_name.json
├── QFS.json
├── timelines.json
└── standard_documents/
```

其中：

- `theme_name.json`：该主题语料库下的全部文档。
- `QFS.json`：包含 query abbreviation、完整 query 描述以及对应的 query-focused reference summary。
- `timelines.json`：原始时间线摘要数据。
- `standard_documents/`：每个 query 对应的相关文档集合及其事件标注。

QFESum 当前包含多个主题事件语料库，例如：

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

具体字段说明、数据格式和样例请参考：

```text
QFESum/README.md
```

## 3. QFES 方法

`QFES/` 文件夹包含本文提出的 QFES 方法及相关实验代码。整个方法主要由两个核心模块组成：

```text
QFES/
├── RAT/
├── SHC/
├── test/
├── unstructured-evidence/
├── graphrag-main/
└── LLM-TLS-main/
```

### 3.1 RAT 检索器

`QFES/RAT/` 是检索器部分。RAT 用于从主题事件语料库中为每个 query 检索相关文档，得到 query-specific relevant document set。

该模块的主要目标是：

1. 计算 query 与文档之间的相关性。
2. 根据自适应阈值选择相关文档。
3. 对灰区文档进行进一步判断。
4. 为后续 SHC 摘要器提供 query-focused 输入文档集合。

具体运行方式请参考：

```text
QFES/RAT/README.md
```

### 3.2 SHC 摘要器

`QFES/SHC/` 是摘要器部分。SHC 基于事件抽取、事件表示、图结构建模和分级聚类进行查询聚焦事件摘要生成。

该模块的主要目标是：

1. 从检索得到的相关文档中抽取 query-focused events。
2. 构建事件图并进行结构建模。
3. 对事件进行聚类、排序和选择。
4. 生成最终的 query-focused event summary。

具体运行方式请参考：

```text
QFES/SHC/README.md
```

### 3.3 测试代码

`QFES/test/` 包含测试代码，用于检查 QFES 方法中不同模块的功能正确性和实验流程完整性。

该部分主要用于：

1. 测试 RAT 检索模块。
2. 测试 SHC 摘要模块。
3. 检查输入输出格式。
4. 验证实验流程是否可以正常运行。

具体运行方式请参考：

```text
QFES/test/README.md
```

## 4. Baseline 方法

本仓库还包含三个 baseline 方法的实验代码，分别位于：

```text
QFES/unstructured-evidence/
QFES/graphrag-main/
QFES/LLM-TLS-main/
```

### 4.1 Unstructured Evidence

`QFES/unstructured-evidence/` 包含基于 evidence extraction 和 LLM summarization 范式的 baseline 实验代码。

该方法首先从输入文档中抽取与 query 相关的 evidence，再基于抽取出的 evidence 生成最终摘要。

具体运行方式请参考：

```text
QFES/unstructured-evidence/README.md
```

### 4.2 GraphRAG

`QFES/graphrag-main/` 包含 GraphRAG baseline 的实验代码。

该方法基于图结构增强的检索与生成流程，用于与本文提出的 QFES 方法进行对比。

具体运行方式请参考：

```text
QFES/graphrag-main/README.md
```

### 4.3 LLM-TLS

`QFES/LLM-TLS-main/` 包含 LLM-TLS baseline 的实验代码。

该方法主要用于基于事件相似性、LLM 语义判断和聚类策略进行时间线或事件摘要生成。

具体运行方式请参考：

```text
QFES/LLM-TLS-main/README.md
```

## 5. 运行环境

本仓库代码在以下环境中测试：

```text
Python 3.11.11
```

建议使用 Conda 创建独立环境：

```bash
conda create -n qfes python=3.11.11
conda activate qfes
```

然后安装总依赖：

```bash
pip install -r requirements.txt
```

如果某个子模块包含独立的 `requirements.txt`，也可以进入对应子目录单独安装：

```bash
cd QFES/RAT
pip install -r requirements.txt
```

或：

```bash
cd QFES/SHC
pip install -r requirements.txt
```

## 6. 基本使用流程

建议按照以下顺序运行实验。

### Step 1: 准备数据集

首先确认 `QFESum/` 数据集已经放置在仓库中，并检查各主题语料库是否包含必要文件：

```text
theme_name.json
QFS.json
timelines.json
standard_documents/
```

### Step 2: 运行 RAT 检索器

进入 RAT 模块：

```bash
cd QFES/RAT
```

按照该模块 README 中的说明运行检索代码。RAT 的输出通常作为 SHC 的输入。

### Step 3: 运行 SHC 摘要器

进入 SHC 模块：

```bash
cd QFES/SHC
```

按照该模块 README 中的说明运行摘要代码，生成 query-focused event summaries。

### Step 4: 运行测试代码

进入测试模块：

```bash
cd QFES/test
```

按照该模块 README 中的说明运行测试代码，检查检索、摘要和评估流程。

## 7. 匿名审稿说明

本仓库用于匿名审稿。代码、文档和元数据中的作者身份信息已被移除。

使用本仓库进行复现或审稿时，请参考各模块下的 README 文件获取详细运行说明。

## 8. License 与数据使用说明

完整的 license 和数据使用说明请参考最终 camera-ready 版本或正式公开版本。