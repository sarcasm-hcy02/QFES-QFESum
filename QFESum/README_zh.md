# QFESum 数据集：结构与使用说明

## 一、概述

QFESum 是一个面向 Query-Focused Event Summarization（查询驱动事件摘要） 的数据集。
该数据集包含多个真实世界重大事件主题，每个主题下均提供：

* 面向查询筛选的文档集合
* 从文档中抽取的事件信息
* 高质量参考摘要

当前数据集包含 8 个主题：

* `bpoil`（BP石油泄漏）
* `finan`（全球金融危机）
* `iraq`（伊拉克战争）
* `syria_t17`(叙利亚战争)
* `egypt`（埃及革命）
* `libya`（利比亚内战）
* `syria_crisis`（叙利亚战争）
* `yemen`（也门革命）

所有主题均采用统一的数据组织结构和格式。

---

## 二、目录结构

以 `bpoil/` 为例，每个主题目录下包含 四个部分：

```id="1tq8r5"
bpoil/
├── bpoil.json
├── standard_documents/
├── QFS.json
└── timelines.json
```

---

## 三、文件说明

### 1. `bpoil.json` —— 全量文档集合

该文件包含该主题下的所有原始文档：

* 未经过查询筛选的完整语料
* 可作为检索或摘要任务的输入来源

---

### 2. `standard_documents/` —— 查询级文档与事件标注

该文件夹包含**按查询划分的标注数据**：

* 每个 `.json` 文件对应一个查询（query）
* 文件名为该查询的缩写形式（见 `QFS.json`）

每个文件中包含：

* 与该查询相关的文档集合
* 从文档中抽取的**查询相关事件信息**

这是数据集中最核心的部分，用于训练和评估 QFES 模型。

---

### 3. `QFS.json` —— 查询定义与参考摘要

该文件提供每个查询的完整信息：

* `query`：查询的完整自然语言描述
* `abbreviation`：查询缩写（对应文件名）
* `reference_summary`：该查询的参考摘要（gold summary）

👉 用于建立：

* 查询文本 ↔ 缩写 ↔ 参考摘要 的映射关系

---

### 4. `timelines.json` —— 原始时间线摘要

该文件包含原始数据集中提供的时间线式摘要：

* 按时间顺序组织的主题级摘要
* 不针对具体查询
* 可用于与 timeline summarization 方法对比

---

## 四、数据使用方法

### Step 1：选择主题

选择一个主题（如 `bpoil`）

---

### Step 2：读取查询信息

从 `QFS.json` 中获取：

* 查询文本（query）
* 查询缩写（abbreviation）
* 参考摘要（reference summary）

---

### Step 3：加载查询对应数据

根据缩写，从：

```
standard_documents/{abbreviation}.json
```

中获取：

* 与该查询相关的文档集合
* 对应的事件信息

---

### Step 4：生成摘要并评估

基于：

* 查询（query）
* 查询相关文档/事件

生成摘要，并与：

* `QFS.json` 中的 `reference_summary` 进行对比评估

---

## 五、注意事项

* 所有主题的数据结构完全一致，便于统一实验
* 查询缩写在 `QFS.json` 中唯一对应
* 事件标注是面向查询的（query-aware），而非全局标注

---

## 六、总结

QFESum 数据集提供：

* 多主题事件语料
* 查询级文档筛选
* 事件级结构化信息
* 高质量参考摘要

适用于以下研究方向：

* 查询驱动摘要（Query-Focused Summarization）
* 检索增强生成（Retrieval-Augmented Generation）
