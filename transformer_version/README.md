# English–French Neural Translator

A **sequence-to-sequence (Seq2Seq)** transformer built with Hugging Face’s `transformers` library to translate English sentences into French.  
This project demonstrates a full **machine translation workflow** — from preprocessing and fine-tuning to deployment with Streamlit.

---

## 1. Overview

The project fine-tunes the pretrained **Helsinki-NLP/opus-mt-en-fr** model on a custom subset of the [Tatoeba English–French corpus](https://opus.nlpl.eu/Tatoeba.php).

It includes:
- Data cleaning and validation
- Tokenisation using `AutoTokenizer`
- Fine-tuning with `Seq2SeqTrainer`
- Model saving for inference
- Streamlit front-end for interactive translation

---

## 2. Repository Structure

```
project/
├── config.py               # Configuration (e.g. n_corpus_lines)
├── train.py                # Data prep, model fine-tuning
├── inference.py            # Translation inference logic
├── app.py                  # Streamlit interface
├── language_pair/
│   └── fra-eng/fra.txt     # Raw bilingual dataset
├── model/                  # Saved model + tokenizer
├── logs/                   # Training logs
└── README.md               # Project documentation
```

---

## 3. Setup

### Prerequisites
Python ≥ 3.10 and `pip` installed.

### Installation

```bash
pip install torch transformers datasets scikit-learn pandas streamlit
```

(Optional, for GPU):

```bash
pip install accelerate
```

---

## 4. Data

The dataset (`fra.txt`) contains **tab-separated English–French pairs**:

```
I am cold.	J'ai froid.
Where are you?	Où es-tu ?
```

The script samples `n_corpus_lines` lines for faster experimentation.

---

## 5. Training Pipeline

Run:

```bash
python train.py
```

### Key steps:
1. **Load & clean** bilingual data  
   - Removes malformed or empty pairs.
2. **Split** into train/validation/test using `train_test_split`.
3. **Tokenise** both source and target sentences.
4. **Fine-tune** the transformer with `Seq2SeqTrainer`.
5. **Save** model and tokenizer for inference in `./model/final`.

Training arguments (in `train.py`):
- `learning_rate=3e-5`
- `num_train_epochs=3`
- `warmup_ratio=0.05`
- `per_device_train_batch_size=8`
- `predict_with_generate=True`

---

## 6. Inference

Translate English text directly in Python:

```python
from inference import translate

text = "This is an example sentence in English."
translation = translate(text)
print(translation)
```

**Output:**
```
Ceci est une phrase d'exemple en anglais.
```

---

## 7. Streamlit App

Run the app locally:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.  
The interface lets you enter English text and view its French translation instantly.
