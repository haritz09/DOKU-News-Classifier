# DOKU — English News Classifier: Fine-Tuning, Prompting & Data Augmentation

A three-part NLP study on classifying BBC news articles into 5 categories (business, entertainment, politics, sport, tech), comparing transformer fine-tuning against LLM prompting, and testing whether LLM-generated synthetic data can substitute real labeled data.

## Part 1 — Fine-tuning transformer models

Fine-tuned **BERT**, **DistilBERT**, and **RoBERTa** for sequence classification on the [SetFit/bbc-news](https://huggingface.co/datasets/SetFit/bbc-news) dataset.

- Hyperparameter optimization via **Bayesian sweeps in Weights & Biases** (learning rate, batch size, epochs, weight decay, warmup ratio).
- Evaluated with accuracy, precision, recall, and weighted F1 (`evaluate` library).
- Compared the three fine-tuned models on a held-out test set to select the best performer.

## Part 2 — Prompt engineering with a quantized LLM

Used **Qwen2.5-1.5B-Instruct**, 4-bit quantized with `bitsandbytes`, to classify articles purely through prompting, no fine-tuning.

- **Zero-shot**: 89.4% accuracy with no examples at all.
- **Multiple-choice** prompting as an alternative framing.
- **Few-shot** (1/3/5 examples): using full articles as examples performed *worse* than zero-shot. Diagnosed the cause (long, noisy examples) and re-ran the experiment with short, concrete example sentences instead — which improved results as more shots were added:

| Shots | F1 (short examples) |
|---|---|
| 1-shot | 0.852 |
| 3-shot | 0.864 |
| 5-shot | 0.902 |

## Part 3 — LLM-generated synthetic data augmentation

Tested whether generating synthetic training examples with an LLM (Qwen2.5) can compensate for small labeled datasets. Fine-tuned DistilBERT on original vs. augmented versions of 50/250/500-example datasets:

| Dataset | Accuracy | F1 |
|---|---|---|
| 50 original | 0.870 | 0.865 |
| 250 original | 0.964 | 0.964 |
| 500 original | 0.970 | 0.970 |
| 200 (50 real + 150 synthetic) | 0.912 | 0.911 |
| 500 (250 real + 250 synthetic) | 0.972 | 0.972 |
| 1000 (500 real + 500 synthetic) | 0.978 | 0.978 |

**Finding:** synthetic augmentation helps most in the smallest-data regime, but doesn't fully substitute real data — 250 real examples alone (96.4%) already beat 200 augmented examples (91.2%).

## Tech stack

Python, PyTorch, Hugging Face (Transformers, Datasets, Evaluate), Weights & Biases, bitsandbytes, scikit-learn, Google Colab

## How to run

Open `DOKU.ipynb` in Google Colab or Jupyter with a GPU runtime. Cells are organized in three independent sections (Z1, Z2, Z3) and run top to bottom within each section.

## Context

Academic NLP project.
