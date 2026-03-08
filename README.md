# seq2seq-en-fr

An end-to-end English–French neural machine translation project, 
built in two stages: first with a custom LSTM encoder-decoder 
architecture, then rebuilt using a fine-tuned Hugging Face 
transformer for significantly stronger results.

The project demonstrates a full machine translation workflow — 
data preprocessing, model training, evaluation against a baseline, 
and an interactive Streamlit inference app.

---

## Results (Transformer)

Evaluated on 500 held-out sentences from the Tatoeba corpus:

| Metric | Fine-tuned | Base model |
|--------|-----------|------------|
| BLEU | 52.11 | 52.44 |
| chrF | 68.12 | 67.64 |
| Avg latency (ms) | 210.5 | 133.5 |

The fine-tuned transformer matches the Helsinki-NLP/opus-mt-en-fr 
base model on BLEU while outperforming it on chrF — the more 
morphologically sensitive metric. See the transformer README for 
full evaluation details and example translations.

---

## Repository Structure
```
seq2seq-en-fr/
├── transformer_version/    # Hugging Face fine-tuned transformer
│   └── README.md           # Full details, setup and results
├── lstm_version/           # Custom LSTM encoder-decoder (v1)
│   └── README.md           # Architecture details and learnings
└── README.md               # This file
```

---

## Motivation

This project grew out of a deliberate progression — starting with 
a hand-built LSTM to understand the mechanics of sequence-to-sequence 
learning, then moving to a transformer architecture to see the 
improvement modern pretraining brings. The LSTM version is preserved 
in this repo as a record of that learning process.

---

## Tech Stack

- Python, PyTorch, Hugging Face Transformers
- Seq2SeqTrainer, AutoTokenizer
- Streamlit (inference UI)
- sacrebleu, chrF (evaluation)

---

## Versions

- **v1 — LSTM**: Custom encoder-decoder with GRU/LSTM cells, 
  embedding layers, trained from scratch on Tatoeba pairs
- **v2 — Transformer**: Fine-tuned Helsinki-NLP/opus-mt-en-fr 
  on 50,000 sentence pairs, evaluated with BLEU and chrF