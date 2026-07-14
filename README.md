# 🧠 LLM From Scratch — GPT Implementation, Pretraining, and Fine-Tuning

A from-first-principles implementation of a GPT-style language model in PyTorch — architecture, pretraining, weight loading from OpenAI's GPT-2, classification fine-tuning, and instruction fine-tuning with LLM-as-judge evaluation.

Every component (attention, transformer blocks, layer norm, sampling, training loops) is written from scratch. No `transformers` library.

---

## 📊 Results at a Glance

| Task | Model | Result |
|---|---|---|
| **Pretraining** (from scratch) | GPT-124M | Val loss 9.93 → 6.45 · overfitting diagnosed on 5K-token corpus |
| **Spam classification** (fine-tuned) | GPT-2 small (124M) | **96.0% test accuracy** (baseline: 48.8%) |
| **Instruction following** (fine-tuned) | GPT-2 medium (355M) | Val loss 0.75 → **0.15** · **50.05/100** (Llama-3-8B as judge) |

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

### Ch 2–4 — Foundations
Tokenization (BPE via `tiktoken`), sliding-window dataset generation, embeddings, self-attention → causal attention → multi-head attention, transformer blocks, and the full GPT architecture.

### Ch 5 — Pretraining & Text Generation

Pretrained the 124M model on *The Verdict* (20,479 chars / 5,145 tokens), 90/10 train/val split, AdamW (lr=4e-4, weight_decay=0.1), 10 epochs.

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 1.98 | 9.93 |
| 5 | 0.77 | 6.16 |
| 10 | **0.08** | **6.45** ⚠️ |

**Diagnosis:** train loss collapsed to 0.08 while val loss bottomed at ~6.13 and then *rose* — textbook overfitting. Verified by memorization: the model reproduced verbatim phrases from the training text ("quite insensible to the irony"). Expected with a 124M-parameter model on 5K tokens.

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

Fine-tuned GPT-2 medium (355M) on 1,100 instruction–response pairs (Alpaca prompt style), split 935/55/110.

**Key implementation details:**
- **Dynamic per-batch padding** via a custom collate function — pads to the longest sequence *in each batch* rather than a global max, cutting wasted compute
- **`-100` masking** on padding targets so they're excluded from cross-entropy loss, while retaining one `<|endoftext|>` so the model learns to terminate

| | Val Loss |
|---|---|
| Before fine-tuning | 0.746 |
| **After 2 epochs** | **0.149** |

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
│   ├── ch2_text_data.ipynb            # Tokenization, embeddings
│   ├── ch3_coding attention...ipynb   # Self → causal → multi-head attention
│   ├── ch4_Implementing a GPT...ipynb # Full GPT architecture
│   ├── ch5_Pretraining.ipynb          # Pretraining, sampling, GPT-2 weight loading
│   ├── ch6_Fine-tuning for classification.ipynb
│   └── ch7-Fine-tuning to follow instructions.ipynb
├── src/
│   ├── Attention.py                   # GPTModel, MultiHeadAttention, training utils
│   └── gpt_download.py                # OpenAI GPT-2 checkpoint loader
├── data/
└── images/
```

---

## 💡 Key Takeaways

- **Pretraining a 124M model on 5K tokens is a memorization exercise, not a learning one.** The val loss curve makes this unmissable — useful as a hands-on demonstration of why scale matters.
- **Selective unfreezing beats full fine-tuning for classification.** Freezing everything except the head, the last transformer block, and the final LayerNorm reached 96% test accuracy in 17 minutes.
- **Loss is not enough for instruction-tuned models.** Val loss dropped 5× (0.75 → 0.15), but the LLM-judge score was only 50/100 — a low loss on next-token prediction does not guarantee good instruction following. This gap is the reason RLHF and preference optimization exist.

---

## ⚠️ Limitations

- Pretraining corpus is tiny (5K tokens) — intentionally, to keep it laptop-runnable
- Instruction fine-tuning stopped at 2 epochs; no hyperparameter search
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

Architecture and curriculum follow **[Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)** by Sebastian Raschka (Manning). All implementations, experiments, and analysis in this repo are my own.

---

## 👤 Author

**Ali Alavi** — M.Sc. Computer Science, University of Messina
[LinkedIn](https://www.linkedin.com/in/ali-alavi-cs/) · [GitHub](https://github.com/salavii)
