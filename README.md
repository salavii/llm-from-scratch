# 🧠 LLM From Scratch — GPT Implementation, Pretraining, and Fine-Tuning

A from-first-principles implementation of a GPT-style language model in PyTorch — architecture, pretraining, weight loading from OpenAI's GPT-2, classification fine-tuning, and instruction fine-tuning with LLM-as-judge evaluation.

Every component (attention, transformer blocks, layer norm, sampling, training loops) is written from scratch. No `transformers` library.

---

## 📊 Results at a Glance

| Task | Model | Result |
|---|---|---|
| **Pretraining** (from scratch) | GPT-124M | Val loss 10.98 → **6.12** (best, epoch 6) then rising · overfitting diagnosed on a 5K-token corpus |
| **Spam classification** (fine-tuned) | GPT-2 small (124M) | **96.0% test accuracy** (baseline: 48.8%) |
| **Instruction following** (fine-tuned) | GPT-2 medium (355M) | Val loss 3.82 → **0.63** · **50.05/100** (Llama-3-8B as judge) |

> **How these numbers were produced.** An earlier version of this repo averaged validation loss incorrectly (see [Key Takeaways](#-key-takeaways)), so the reported losses have been regenerated — by two different routes, worth distinguishing:
>
> - **Ch5** is a **fresh 10-epoch run** on the current code (RTX 3050, 4 GB). It is not a replay of the original run: the LayerNorm epsilon fix shifts the trajectory slightly, so early epochs track the original closely and later ones drift.
> - **Ch7** is the **original fine-tuned checkpoint re-measured**, not a retrain. The bug lived in `calc_loss_loader`, which only ever fed evaluation logging — `train_model_simple` backpropagates through `calc_loss_batch` directly, so gradients were never affected and the saved weights are exactly what the original run produced. Re-evaluating recovers the numbers that run should have printed. (Loading that checkpoint under the fixed LayerNorm changes val loss by 7.8e-05, i.e. nothing.)
> - **Ch6** is untouched. It defined its own correct averaging function, so its accuracies were never wrong.

---

## 🚀 Setup

**Requirements:** Python 3.11+, and a CUDA GPU if you want to train in reasonable time (everything also runs on CPU, more slowly).

```bash
git clone https://github.com/salavii/llm-from-scratch.git
cd llm-from-scratch

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
jupyter lab
```

Each notebook opens with a bootstrap cell that locates the repo root, puts `src/` on the import path, and pins the working directory — so notebooks run correctly no matter where Jupyter was started.

**Run them in order:** `ch1` → `ch2` → `ch3` → `ch4` → `ch5` → `ch6` → `ch7`. Chapters 5–7 each stand alone if you already have the artifacts they need.

**What gets downloaded on first run** (none of it is committed):

| What | Size | Where | Used by |
|---|---|---|---|
| GPT-2 small TF checkpoint | ~500 MB | `gpt2/124M/` | Ch5, Ch6 |
| GPT-2 medium TF checkpoint | ~1.5 GB | `gpt2/355M/` | Ch7 |
| SMS Spam Collection | ~200 KB | `data/sms_spam_collection/` | Ch6 |

Trained checkpoints are written to `checkpoints/` (git-ignored; 0.5–1.7 GB each). They are **not** distributed — rerun the training cells to regenerate them.

**Extra step for Ch7's evaluation only.** The LLM-as-judge section scores responses with Llama 3 8B served locally by [Ollama](https://ollama.com):

```bash
ollama serve
ollama pull llama3:8b
```

`tensorflow` is in `requirements.txt` solely because `src/gpt_download.py` reads OpenAI's original TensorFlow checkpoint format. Skip the GPT-2 weight-loading sections and you can skip that dependency.

---

## 🏗️ Architecture (implemented from scratch)

```
GPTModel (124M params)
├── Token embedding      (50257 × 768)
├── Positional embedding (1024 × 768)
└── 12 × TransformerBlock
    ├── MultiHeadAttention  (12 heads, causal masking, dropout)
    ├── FeedForward         (768 → 3072 → 768, GELU)
    ├── LayerNorm × 2       (custom implementation)
    └── Residual connections
└── Final LayerNorm + output head (768 → 50257)
```

Configurable across all GPT-2 sizes: 124M / 355M / 774M / 1558M.

---

## 📖 Chapter Breakdown

### Ch 1–4 — Foundations
LLM landscape overview, tokenization (BPE via `tiktoken`), sliding-window dataset generation, embeddings, self-attention → causal attention → multi-head attention, transformer blocks, and the full GPT architecture.

### Ch 5 — Pretraining & Text Generation

Pretrained the 124M model on *The Verdict* (20,479 chars / 5,145 tokens), 90/10 train/val split, AdamW (lr=4e-4, weight_decay=0.1), 10 epochs. That leaves 9 training batches and 1 validation batch at `batch_size=2, context_length=256`.

| Checkpoint | Train Loss | Val Loss |
|---|---|---|
| Initialization | 10.99 | 10.98 |
| Epoch 1 | 9.83 | 9.93 |
| Epoch 5 | 4.22 | 6.18 |
| **Epoch 6** | 3.84 | **6.12** ← best |
| Epoch 10 | 0.77 | 6.37 ⚠️ |

Losses are averaged over `eval_iter=5` batches. Evaluated over the *full* dataset, the final model sits at train **0.59** / val **6.41**.

**Sanity check:** at initialization the loss is 10.98, essentially `ln(50257) = 10.82` — exactly what a uniform distribution over the vocabulary should give. A pretraining run that doesn't start there has a bug.

**Diagnosis:** validation loss bottoms out at 6.12 in epoch 6 and then *climbs* while training loss keeps falling to 0.59 — textbook overfitting. Verified by memorization: the model reproduced verbatim phrases from the training text ("quite insensible to the irony"). Entirely expected for a 124M-parameter model on 5K tokens.

**Also implemented:**
- **Temperature scaling** — logit division before softmax to control randomness
- **Top-k sampling** — masking non-top-k logits to `-inf` before renormalizing
- **GPT-2 weight loading** — manually mapping OpenAI's TensorFlow checkpoint (`c_attn` concatenated QKV, `c_proj`, `ln_1/ln_2`, weight tying) into the custom PyTorch `GPTModel`. Verified correct by coherent generation.

### Ch 6 — Classification Fine-Tuning (Spam Detection)

Fine-tuned GPT-2 small on the SMS Spam Collection (UCI), balanced by undersampling to 747/747, split 70/10/20.

**Approach:** freeze all pretrained weights → replace the 50257-dim output head with a 2-class head → *selectively unfreeze* only the final transformer block and final LayerNorm. Classification uses the **last token's** logits, which under causal attention has visibility of the entire sequence.

| Stage | Train Acc | Val Acc | Test Acc |
|---|---|---|---|
| Before fine-tuning | 46.3% | 45.0% | 48.8% |
| **After 5 epochs** | **96.9%** | **97.3%** | **96.0%** |

Training time: 17 min. Loss dropped 2.45 → 0.09.

### Ch 7 — Instruction Fine-Tuning

Fine-tuned GPT-2 medium (355M) on 1,100 instruction–response pairs (Alpaca prompt style), split 935/55/110. The split is sequential rather than shuffled, following the book.

**Key implementation details:**
- **Dynamic per-batch padding** via a custom collate function — pads to the longest sequence *in each batch* rather than a global max, cutting wasted compute
- **`-100` masking** on padding targets so they're excluded from cross-entropy loss, while retaining one `<|endoftext|>` so the model learns to terminate

| | Train Loss | Val Loss | Test Loss |
|---|---|---|---|
| Before fine-tuning | 3.79 | 3.82 | 3.84 |
| **After 2 epochs** | **0.30** | **0.63** | **0.68** |

**Before fine-tuning** (instruction: convert to passive voice):
> repeats the prompt back, fails to follow the instruction

**After fine-tuning:**
> *"The meal is cooked every day by the chef."* ✅

**Automated evaluation (LLM-as-judge):** scored all 110 test responses using **Llama 3 8B** running locally via Ollama, prompting for an integer 0–100 quality score against the reference answer.

**Average score: 50.05 / 100**

---

## 🛠 Stack

`Python` · `PyTorch` · `tiktoken` · `NumPy` · `Ollama` (Llama 3 8B, evaluation only)

No `transformers`, no `peft` — everything hand-rolled.

---

## 📂 Structure

```
llm-from-scratch/
├── notebooks/
│   ├── ch1_llm_overview.ipynb         # LLM landscape (narrative only)
│   ├── ch2_text_data.ipynb            # Tokenization, embeddings
│   ├── ch3_coding attention...ipynb   # Self → causal → multi-head attention
│   ├── ch4_Implementing a GPT...ipynb # Full GPT architecture
│   ├── ch5_Pretraining.ipynb          # Pretraining, sampling, GPT-2 weight loading
│   ├── ch6_Fine-tuning for classification.ipynb
│   └── ch7-Fine-tuning to follow instructions.ipynb
├── src/
│   ├── Attention.py                   # GPTModel, MultiHeadAttention, training utils
│   └── gpt_download.py                # OpenAI GPT-2 checkpoint loader
├── data/                              # Datasets (see NOTICE for provenance)
├── images/                            # Figures used by the notebooks
├── checkpoints/                       # Trained weights (git-ignored, created on demand)
├── requirements.txt
├── LICENSE                            # Apache 2.0
└── NOTICE                             # Third-party attributions
```

---

## 💡 Key Takeaways

- **Pretraining a 124M model on 5K tokens is a memorization exercise, not a learning one.** The val loss curve makes this unmissable — useful as a hands-on demonstration of why scale matters.
- **Selective unfreezing beats full fine-tuning for classification.** Freezing everything except the head, the last transformer block, and the final LayerNorm reached 96% test accuracy in 17 minutes.
- **Loss is not enough for instruction-tuned models.** Val loss dropped 6× (3.82 → 0.63), but the LLM-judge score was only 50/100 — a low loss on next-token prediction does not guarantee good instruction following. This gap is the reason RLHF and preference optimization exist.
- **Always sanity-check your metrics against a known baseline.** An earlier version of this repo averaged validation loss incorrectly, which quietly reported the pretraining loss as starting near 1.2 instead of ~11. Comparing against `ln(vocab_size)` catches that class of bug instantly.

---

## ⚠️ Limitations

- Pretraining corpus is tiny (5K tokens) — intentionally, to keep it laptop-runnable
- Instruction fine-tuning stopped at 2 epochs; no hyperparameter search
- Ch7's train/val/test split is sequential, not shuffled
- LLM-as-judge (Llama 3 8B) is a noisy evaluator; a larger judge or human eval would be more reliable
- No RLHF / DPO stage

---

## 🔮 Future Work

- Attention weight visualizations
- Nucleus (top-p) sampling and beam search
- DPO / preference optimization on top of the SFT model
- Gradio demo for the instruction-tuned model

---

## 📚 Reference

Architecture and curriculum follow **[Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)** by Sebastian Raschka (Manning). The experiments, analysis, and results in this repo are my own; `src/gpt_download.py` is taken essentially verbatim from the book's companion repository under Apache 2.0. See [NOTICE](NOTICE) for full third-party attribution, including dataset provenance.

## 📄 License

[Apache License 2.0](LICENSE).

---

## 👤 Author

**Ali Alavi** — M.Sc. Computer Science, University of Messina
[LinkedIn](https://www.linkedin.com/in/ali-alavi-cs/) · [GitHub](https://github.com/salavii)
