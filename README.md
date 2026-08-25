# CommonsenseQA – Transformers & LLM Approaches (Individual Project 2)

An individual NLP project evaluating commonsense reasoning on the **CommonsenseQA** benchmark using fine-tuned **BERT (Random vs. Pretrained)** and **Google Flan-T5-XL (Zero-Shot vs. Few-Shot)**.

---

## Project Context
- **Author:** Flora Gashi (Hochschule Luzern)
- **Project Type:** Individual NLP Assignment (Project 2)
- **Goal:** Compare fine-tuning encoder-based transformers against zero-shot/few-shot prompting of large language models for commonsense reasoning.

---

## Pipeline & Preprocessing
- **Tokenizer:** `bert-base-uncased` tokenizer with space cleaning.
- **Formatting:** Preserved punctuation, case preservation via lowercasing, and kept all stop words / tokens intact.
- **Inputs:**
  - **BERT:** Formatted as 5 pairs per question: `[CLS] Question [SEP] Answer_i [SEP]` $\rightarrow$ 12-layer encoder $\rightarrow$ `[CLS]` embedding ($768d$) $\rightarrow$ Linear Classifier.
  - **Flan-T5-XL:** Formatted prompt listing options A–E, prompting for direct text generation.

---

## Experiments & Hyperparameter Setup

### BERT Fine-Tuning Setup
Grid search over **12 runs per model** using CrossEntropyLoss, AdamW, and Batch Size 64:
- **Learning Rates:** `1e-3`, `1e-4`, `1e-5`
- **Weight Decay:** `0.01`, `0.1`
- **Schedulers:** Linear, Cosine
- **Best Configurations:**
  - **Random Init BERT:** LR `1e-4`, Weight Decay `0.01`.
  - **Pretrained BERT:** LR `1e-3`, Cosine Scheduler, Weight Decay `0.01`.

### Flan-T5-XL LLM Setup (2.85B Parameters)
- **Zero-Shot:** Direct instruction prompting.
- **Few-Shot:** 3 in-context dataset examples provided within the prompt.

---

## Results & Comparative Analysis

| Model | Setup | Validation Accuracy | Test Accuracy |
|-------|-------|---------------------|---------------|
| **Random Baseline** | Floor | 20.0% | 20.0% |
| **BERT (Random Init)** | Fine-tuned | 17.4% | **22.1%** |
| **BERT (Pretrained)** | Fine-tuned | 57.2% | **55.2%** |
| **Flan-T5-XL** | Zero-Shot | - | **88.7%** |
| **Flan-T5-XL** | Few-Shot (3-shot) | - | **89.2%** |

**Presentation:** [Download Presentation PDF](./presentation_transformers.pdf)

---

## Key Findings & Error Analysis

- **Pretraining is Mandatory:** Randomly initialized BERT barely performed better than random guessing ($22.1\%$), whereas Pretrained BERT jumped to $55.2\%$ test accuracy.
- **Prompting > Fine-tuning Small Encoders:** Large scale pretrained LLMs (Flan-T5-XL) vastly outperformed fine-tuned BERT ($89.2\%$ vs $55.2\%$) even with minimal prompting.
- **Error Patterns:** 
  - BERT models suffered from predicting semantically misleading options.
  - In total, **379 questions** were answered correctly by Flan-T5-XL where both BERT models failed.
- **Question Prefixes:** Questions starting with *"if"* and *"when"* saw the highest accuracy improvements under Pretrained BERT.

---

## How to Run

1. Clone the repo & install requirements:
   ```bash
   git clone TODO
   cd commonsense-qa-transformers
   pip install -r requirements.txt
