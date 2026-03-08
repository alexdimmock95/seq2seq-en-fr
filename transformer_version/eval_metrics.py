# ---------- Imports ----------
import random
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from evaluate import load

# ---------- Config ----------
DATA_PATH = "./data/fra.txt"
MODEL_PATH = "./model"   # fine-tuned model
BASE_MODEL_PATH = "Helsinki-NLP/opus-mt-en-fr"  # base model for comparison
N_CORPUS_LINES = 50000
N_EVAL_SAMPLES = 500           # number of test sentences to evaluate on

# ---------- Load Data (same logic as training.py) ----------
print("Loading data...")
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

# Use a fixed seed here so results are reproducible
random.seed(42)
lines = random.sample(lines, N_CORPUS_LINES)

pairs = [line.split('\t')[:2] for line in lines]
clean_pairs = [p for p in pairs if len(p) == 2 and all(
    isinstance(x, str) and x.strip() != "" for x in p
)]

df = pd.DataFrame(clean_pairs, columns=["en", "fr"])

# Recreate the same train/test split as training.py
train_df_temp, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Sample from test set for evaluation
eval_df = test_df.sample(n=min(N_EVAL_SAMPLES, len(test_df)), random_state=42)
source_sentences = eval_df["en"].tolist()
reference_sentences = eval_df["fr"].tolist()

print(f"Evaluating on {len(source_sentences)} sentences from held-out test set\n")

# ---------- Load Models ----------
print("Loading fine-tuned model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
model.eval()

print("Loading base model for comparison...")
base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_PATH)
base_model.eval()

# ---------- Translate Function ----------
def translate_batch(sentences, tok, mod, max_length=64):
    """Translate a list of sentences, returning translations and avg latency."""
    translations = []
    latencies = []
    for sentence in sentences:
        start = time.time()
        inputs = tok(sentence, return_tensors="pt",
                     max_length=max_length, truncation=True)
        outputs = mod.generate(**inputs, max_length=max_length)
        translation = tok.decode(outputs[0], skip_special_tokens=True)
        latencies.append((time.time() - start) * 1000)  # convert to ms
        translations.append(translation)
    avg_latency = sum(latencies) / len(latencies)
    return translations, avg_latency

# ---------- Run Evaluation ----------
print("Translating with fine-tuned model (this may take a few minutes)...")
ft_translations, ft_latency = translate_batch(
    source_sentences, tokenizer, model
)

print("Translating with base model for comparison...")
base_translations, base_latency = translate_batch(
    source_sentences, base_tokenizer, base_model
)

# ---------- Compute Metrics ----------
print("\nComputing BLEU and chrF scores...")
bleu_metric = load("sacrebleu")
chrf_metric = load("chrf")

# sacrebleu expects references as list of lists
references_wrapped = [[ref] for ref in reference_sentences]

ft_bleu = bleu_metric.compute(
    predictions=ft_translations,
    references=references_wrapped
)
ft_chrf = chrf_metric.compute(
    predictions=ft_translations,
    references=reference_sentences
)

base_bleu = bleu_metric.compute(
    predictions=base_translations,
    references=references_wrapped
)
base_chrf = chrf_metric.compute(
    predictions=base_translations,
    references=reference_sentences
)

# ---------- Print Results ----------
print("\n" + "="*55)
print("EVALUATION RESULTS")
print("="*55)
print(f"{'Metric':<25} {'Fine-tuned':>12} {'Base model':>12}")
print("-"*55)
print(f"{'BLEU score':<25} {ft_bleu['score']:>12.2f} {base_bleu['score']:>12.2f}")
print(f"{'chrF score':<25} {ft_chrf['score']:>12.2f} {base_chrf['score']:>12.2f}")
print(f"{'Avg latency (ms)':<25} {ft_latency:>12.1f} {base_latency:>12.1f}")
print(f"{'Test samples':<25} {len(source_sentences):>12}")
print("="*55)

# ---------- Show Example Translations ----------
print("\nEXAMPLE TRANSLATIONS (5 random samples)")
print("-"*55)
sample_indices = random.sample(range(len(source_sentences)), 5)
for i in sample_indices:
    print(f"\nSource:     {source_sentences[i]}")
    print(f"Reference:  {reference_sentences[i]}")
    print(f"Fine-tuned: {ft_translations[i]}")
    print(f"Base model: {base_translations[i]}")

print("\n✅ Done. Copy the results table above into your README.")