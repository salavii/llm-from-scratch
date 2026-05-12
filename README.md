# 🧠 LLM From Scratch

A personal implementation-focused repository for understanding GPT-style Large Language Models from first principles using PyTorch.

This project documents my learning process while studying transformer-based language models, including tokenization, attention mechanisms, GPT architecture, pretraining, and text generation.

---

## 📌 Project Goals

- Understand how GPT-style architectures work internally
- Implement transformer components from scratch
- Explore tokenization, embeddings, self-attention, and training workflows
- Learn practical LLM pretraining and inference pipelines

---

## 🧱 Topics Covered

- Tokenization and vocabulary creation
- Sliding-window dataset generation
- Embeddings and positional encoding
- Self-Attention and Multi-Head Attention
- Transformer blocks
- GPT-style autoregressive generation
- Training and validation loops
- Loss evaluation and text generation
- Loading pretrained GPT-2 weights

---

## 📂 Repository Structure

```text
llm-from-scratch/
│
├── notebooks/
│   ├── ch1_llm_overview.ipynb
│   ├── ch2_text_data.ipynb
│   └── ch5_Pretraining.ipynb
│
├── src/
│   ├── Attention.py
│   └── gpt_download.py
│
├── data/
│   ├── instruction-data.json
│   ├── instruction-data-with-response.json
│   ├── the-verdict.txt
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
│
├── images/
│   ├── accuracy-plot.pdf
│   └── loss-plot.pdf
│
├── .gitignore
└── README.md
```

---

## 📓 Included Notebooks

| Notebook | Description |
|---|---|
| `ch1_llm_overview.ipynb` | Introduction to LLMs and GPT architecture |
| `ch2_text_data.ipynb` | Text preprocessing, tokenization, and vocabulary creation |
| `ch5_Pretraining.ipynb` | GPT pretraining workflow, logits, loss calculation, and generation |

---

## ⚙️ Core Implementation

Main implementation modules include:

- Multi-Head Self-Attention
- Transformer Blocks
- GPT-style autoregressive model
- Custom Layer Normalization
- Text generation utilities
- GPT-2 weight loading utilities
- Training and evaluation pipeline

Implemented using:

- Python
- PyTorch
- NumPy
- tiktoken

---

## 📚 Learning Source

This repository is heavily inspired by:

**Build a Large Language Model (From Scratch)**  
Sebastian Raschka — Manning Publications

This repository contains my own implementation notes, experiments, and learning exercises while studying the book.

---

## 🚀 Future Improvements

- Add attention visualizations
- Add sampling strategies (top-p / beam search)
- Improve training monitoring
- Add inference demo with Gradio or Streamlit
- Train on larger datasets

---

## 👤 Author

Ali Alavi
