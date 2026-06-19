# The Fine-Tuned BERT with One-Class Ensemble (BERRT-OCEn)

---

## Description

This repository implements a **three-stage anomaly detection pipeline** for textual data:

1. **Domain-adaptive BERT fine-tuning**  
   Adapts a pre-trained BERT model to the target domain using only normal-class samples.

2. **One-Class Classification with CVDD**  
   Uses the **Context Vector Data Description (CVDD)** algorithm with multi-head self-attention to learn normal semantic patterns.

3. **Ensemble Integration**  
   Combines multiple one-class detectors via **stacking** or **max-voting** strategies to improve robustness and detection performance.

The framework is designed to detect **fund misuse, irregular expenditures, and anomalous administrative text records**.

---

## Dataset Information

### Provided Test Dataset

A small test dataset is included for validation and format reference:  
`data/corpora/test.xlsx`

### Supported Public Datasets

The codebase is compatible with standard benchmark datasets:

- **Reuters-21578**  
  https://www.daviddlewis.com/resources/testcollections/reuters21578/
- **20 Newsgroups**  
  http://qwone.com/~jason/20Newsgroups/
- **IMDB Movie Reviews**  
  https://ai.stanford.edu/~amaas/data/sentiment/

### Non-Public Institutional Datasets

The following datasets originate from a government agency and have been anonymized:

- **Dataset1**: `data/corpora/yixing.xlsx`
- **Dataset2**: `data/corpora/nanchang.xlsx`

### Dataset Schema

All datasets share the same schema:

| Column       | Description          |
|-------------|----------------------|
| `index`     | Sample identifier    |
| `Purpose`   | Input text           |
| `Category`  | Class label          |

---

## Code Information

The project is organized as follows:

```
├── src/
│   ├── main.py                  # CLI entry point
│   ├── datasets/                # Dataset loaders
│   ├── networks/                # CVDD & BERT modules
│   ├── baselines/               # OC-SVM baseline
│   ├── ensemble/                # Ensemble methods
│   └── utils/                   # Utilities & visualization
├── data/
│   ├── corpora/                 # Raw datasets
│   └── bert_cache/              # BERT model cache
├── result/                      # Experimental outputs
└── requirements.txt
```

Key components:

- **CVDDNet**: Self-attention-based one-class detector
- **BERT Embedding**: Domain-adaptive fine-tuning
- **OC-SVM**: Baseline comparison
- **Ensemble Module**: Stacking & max aggregation

---

## Requirements

- Python **3.7**
- PyTorch >= 1.7
- CUDA-enabled GPU (recommended)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Installation

This code is written in `Python 3.7` and requires the packages listed in `requirements.txt`.

To run the code, we recommend setting up a virtual environment, e.g. using `conda`:

```bash
cd <path-to-BERT-OCEn-directory>
conda create --name myenv
source activate myenv
pip install -r requirements.txt
```

---

## Usage Instructions

### Running Experiments

You can run BERT-OCEn experiments using the `main.py` script.

Next, I will briefly explain how to run the `test` dataset. You can run other datasets by replacing the dataset file read by `pandas` in `src/main.py` and `src/datasets/*.py`.

#### Example: `test` Dataset (Chinese BERT)

```bash
cd <path-to-BERT-OCEn-directory>

# activate virtual environment
source myenv/bin/activate  # or 'source activate myenv' for conda

# change to source directory
cd src

# create folder for experimental output
mkdir ../log/test_yixing

# run experiment
python main.py yixing cvdd_Net ../log/test_yixing ../data \
  --device cuda \
  --seed 1 \
  --clean_txt \
  --embedding_size 768 \
  --pretrained_model bert \
  --ad_score context_dist_mean \
  --n_attention_heads 4 \
  --attention_size 300 \
  --lambda_p 1.0 \
  --alpha_scheduler logarithmic \
  --n_epochs 100 \
  --lr 0.01 \
  --lr_milestone 40 \
  --train_proportion 0.01 \
  --finetune 1 \
  --normal_class 100 \
  --ensemble stacking
```

Have a look into `main.py` for all the possible arguments and options.

---

## Methodology

### 1. BERT Fine-Tuning

- Only normal-class samples are used
- Last transformer layers are updated
- Domain adaptation improves semantic representation

### 2. CVDD One-Class Learning

- Multi-head self-attention extracts contextual features
- Multiple context vectors represent normal semantics
- Anomaly score = distance to context vectors

### 3. Ensemble Strategy

Two ensemble modes are supported:

- **Stacking**: Meta-classifier over detector outputs
- **Max**: Maximum anomaly score across detectors

---

## Citations

If you use this codebase or datasets in your research, please cite:

```bibtex
@article{ruff2019self,
  title={Self-Attention-Based Anomaly Detection in Text},
  author={Ruff, Lukas and others},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2019}
}
```

---

## License

This project is licensed under the **MIT License**.  
See `LICENSE` for details.

---

## Contribution Guidelines

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear documentation

For issues or questions, please open a GitHub issue.


