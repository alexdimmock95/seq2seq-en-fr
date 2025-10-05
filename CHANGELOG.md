# Changelog

## 25-10-05 Commit 5 (Transformer Model)
- Recreated pipeline with HF Transformer architecture. Includes dataset cleaning and preprocessing, automatic tokenisation, seq2seq training adding to pre-trained model, and inference. Inference wrapped in Streamlit front-end.

## 2025-10-01 Commit 4
- Introduced Luong attention mechanism. Observed far better translation results with 256 latent dim and 10k input line size.
- Replaced regex based tokenisation with Hugging Face BPE (Byte Pair Encoding). Removes need for creation of input and target dictionaries. Found worse translation predictions with smaller tokens, improved with parameter tweaking.
- Introduced early stopping and dynamic learning rate to optimise training time and quality.

## 2025-09-21 Commit 3
- Added detailed comments for all lines of code. Primarily for personal learning but also to practise commenting all code. Future projects won't be as verbose. 

## 2025-09-17 Commit 2
- Integrated embedding layers into LSTM model for semantic representation and memory efficiency.
- Explored embedding dimensions 50, 100, 200.

## 2025-09-15 Initial Commit (LSTM Model)
- Initial seq2seq LSTM model implemented.
- Trained on 10-20k sentence pairs for validation.