# 🧠 LLM From Scratch

A personal implementation-focused repository for understanding GPT-style Large Language Models from first principles using PyTorch.

This project documents my learning process while studying transformer-based language models, including tokenization, attention mechanisms, GPT architecture, pretraining, fine-tuning, and text generation.

---

## 📌 Project Goals

- Understand how GPT-style architectures work internally
- Implement transformer components from scratch
- Explore tokenization, embeddings, self-attention, and training workflows
- Learn practical LLM pretraining, fine-tuning, and inference pipelines

---

## 🧱 Topics Covered

- Tokenization and vocabulary creation
- Sliding-window dataset generation
- Embeddings and positional encoding
- Self-Attention, Causal Attention, and Multi-Head Attention
- Transformer blocks
- GPT-style autoregressive generation
- Training and validation loops
- Loss evaluation and text generation
- Classification fine-tuning
- Instruction fine-tuning
- Loading pretrained GPT-2 weights

---

## 📂 Repository Structure

```text
llm-from-scratch/
│
├── notebooks/
│   ├── ch1_llm_overview.ipynb
│   ├── ch2_text_data.ipynb
│   ├── ch3_coding attention mechanisms.ipynb
│   ├── ch4_Implementing a GPT model from scratch to generate text.ipynb
│   ├── ch5_Pretraining.ipynb
│   ├── ch6_Fine-tuning for classification.ipynb
│   └── ch7-Fine-tuning to follow instructions.ipynb
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
| [ch1_llm_overview.ipynb](notebooks/ch1_llm_overview.ipynb) | Introduction to LLMs, transformers, and GPT architecture |
| [ch2_text_data.ipynb](notebooks/ch2_text_data.ipynb) | Text preprocessing, tokenization, vocabulary creation, and embeddings |
| [ch3_coding attention mechanisms.ipynb](notebooks/ch3_coding%20attention%20mechanisms.ipynb) | Self-attention, causal masking, dropout, and multi-head attention |
| [ch4_Implementing a GPT model from scratch to generate text.ipynb](notebooks/ch4_Implementing%20a%20GPT%20model%20from%20scratch%20to%20generate%20text.ipynb) | GPT model architecture, layer normalization, transformer blocks, and generation |
| [ch5_Pretraining.ipynb](notebooks/ch5_Pretraining.ipynb) | GPT pretraining workflow, logits, loss calculation, and generation |
| [ch6_Fine-tuning for classification.ipynb](notebooks/ch6_Fine-tuning%20for%20classification.ipynb) | Fine-tuning a pretrained GPT-style model for classification |
| [ch7-Fine-tuning to follow instructions.ipynb](notebooks/ch7-Fine-tuning%20to%20follow%20instructions.ipynb) | Instruction fine-tuning workflow and response generation |

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
