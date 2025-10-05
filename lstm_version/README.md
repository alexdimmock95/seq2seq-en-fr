# Seq2Seq English–French Translator  
Neural Machine Translation using a custom BPE tokenizer, LSTM encoder–decoder with Luong Attention, and inference decoding.

---

## Objective

This project implements a complete English → French Neural Machine Translation (NMT) pipeline.  
It evolves from a basic sequence-to-sequence (seq2seq) recurrent model to a fully attention-enabled bilingual system, capable of translating at the sentence level.

---

## Architecture Overview

### 1. Custom Preprocessing and Tokenisation
- Uses a custom Byte-Pair Encoding (BPE) tokenizer built via Hugging Face’s `tokenizers` library.  
- Both English and French corpora are cleaned, tokenised, and trained jointly into a shared subword vocabulary.  
- Special tokens (`[PAD]`, `[UNK]`, `<START>`, `<END>`, etc.) are manually added to handle sequence control.  
- The data is split into training, validation, and test subsets using `scikit-learn`.

### 2. Seq2Seq Model with Luong Attention
The model consists of:

- **Encoder:**  
  - Embedding layer (masking `[PAD]` tokens)  
  - LSTM producing contextual hidden and cell states  

- **Decoder:**  
  - Embedding layer (separate from encoder)  
  - LSTM generating output tokens with teacher forcing  
  - Luong-style attention mechanism implemented via dot product and masking  
  - Dense softmax layer converting context-aware outputs to token probabilities  

**Loss:** `sparse_categorical_crossentropy`  
**Optimiser:** `Adam` (with gradient clipping)  
**Regularisation:** Early stopping and learning rate reduction on plateau  

### 3. Inference Pipeline
Once trained:
- The encoder model extracts latent state vectors (`h`, `c`).  
- The decoder model is restructured for token-by-token generation.  
- Greedy decoding produces translations autoregressively until `<END>` or maximum sequence length.
