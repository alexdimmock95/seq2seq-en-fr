# English–French Neural Translator (LSTM v1)

A sequence-to-sequence (Seq2Seq) encoder–decoder model built from 
scratch to translate English sentences into French, using LSTM and 
GRU recurrent architectures.

This is v1 of the seq2seq-en-fr project — built to understand the 
mechanics of neural machine translation before moving to a 
transformer-based approach in v2.

---

## 1. Overview

The model implements a classic encoder–decoder architecture trained 
on a subset of the Tatoeba English–French corpus. Unlike the 
transformer version, this model is built from scratch rather than 
fine-tuned from a pretrained checkpoint — giving direct insight into 
how sequence-to-sequence learning works at a lower level.

It includes:
- Custom tokenisation and vocabulary building
- Embedding layers for memory-efficient input representation
- Encoder–decoder architecture with LSTM and GRU variants
- Training loop with Adam and RMSprop optimiser experiments
- Qualitative translation evaluation

---

## 2. Repository Structure
```
lstm_version/
├── config.py               # Configuration and hyperparameters
├── preprocessing.py        # Tokenisation and vocabulary building
├── layers.py               # Encoder and decoder model definitions
├── training_model.py       # Training loop
├── test_function.py        # Inference and translation testing
└── README.md               # This file
```

---

## 3. Setup

### Prerequisites
Python ≥ 3.10 and `pip` installed.

### Installation
```bash
pip install torch numpy pandas
```

---

## 4. Data

The dataset uses tab-separated English–French sentence pairs from 
the [Tatoeba corpus](https://opus.nlpl.eu/Tatoeba.php).

Download and place at `lstm_version/language_pair/fra-eng/fra.txt` 
before running.

The model was trained initially on 10–20k sentence pairs to validate 
the approach before scaling up.

---

## 5. Architecture

### Encoder
Takes a tokenised English sentence and produces a fixed-length 
context vector capturing the meaning of the input sequence.

### Decoder
Takes the context vector and generates the French translation 
token by token.

### Embedding layers
Token indices are mapped to dense vectors rather than one-hot 
encodings — significantly more memory-efficient for large 
vocabularies.

### Variants
Both LSTM and GRU cells were tested. GRU provided faster training 
with comparable translation quality for this dataset size.

---

## 6. Training
```bash
python training_model.py
```

Key hyperparameters:
- Optimiser: Adam (also tested RMSprop)
- Trained on subsets of 10k–20k sentence pairs
- Training loss decreased from 1.6 → 1.0 across epochs

---

## 7. Key Learnings

- Encoder–decoder RNNs begin producing recognisable translations 
  after ~10 epochs, with fluency improving as dataset size increases
- GRUs are faster and lighter than LSTMs while achieving comparable 
  quality at this scale
- Embedding inputs are far more memory-efficient than one-hot 
  representations
- Fixed-length context vectors become a bottleneck for longer 
  sentences — the core limitation that attention mechanisms and 
  transformers address

---

## 8. Limitations

The fixed context vector architecture has a fundamental ceiling — 
longer or more complex sentences suffer because all input information 
must be compressed into a single vector before decoding begins. This 
is the key motivation for moving to the transformer architecture in 
v2, where attention allows the decoder to reference any part of the 
input sequence at each generation step.

---

## 9. Next Steps (carried into v2)

- Attention mechanisms (Bahdanau or Luong) to improve longer 
  sequence translation
- Transformer architecture with pretrained weights
- Subword tokenisation via Byte Pair Encoding
- Quantitative evaluation with BLEU and chrF scores

See `transformer_version/` for the v2 implementation.