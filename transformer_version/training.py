# ---------- Imports ----------
from config import n_corpus_lines

import random
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

import pandas as pd

# ---------- Preprocessing ----------
# Import language dataset
data_path = "/Users/Alex/Documents/Coding/2. Data Scientist - Natural Language Processing Specialist/deep_learning_project/language_pair/fra-eng/fra.txt"

# Defining lines as a list of each line
with open(data_path, 'r', encoding='utf-8') as f:   # Open in read mode
  lines = f.read().split('\n')    # Split each line by newline

lines = random.sample(lines, n_corpus_lines)  # Randomly sample n lines from the corpus

# Creates paired lists of input and target sentences
pairs = [line.split('\t')[:2] for line in lines]    # Splits each line at tab, takes index 0 and 1 and returns in list

# Filter out malformed lines
clean_pairs = []
for p in pairs:
    if len(p) == 2 and all(isinstance(x, str) and x.strip() != "" for x in p):
        clean_pairs.append(p)

print(f"✅ Cleaned {len(clean_pairs)} valid pairs out of {len(pairs)} total")
df = pd.DataFrame(clean_pairs, columns=["en", "fr"])

# Split df into train_df_temp and test_df splits
train_df_temp, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Split train_temp into train and val splits
train_df, val_df = train_test_split(train_df_temp, test_size=0.1, random_state=42)

# Transfer split df's into HF Dataset
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "val": Dataset.from_pandas(val_df),
    "test": Dataset.from_pandas(test_df)
})

# ---------- Transformer Model Selection ----------

model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# ---------- Tokenisation ----------

max_length = 64  # cap to avoid long sequences

def tokenisation(batch):
    model_inputs = tokenizer(batch["en"], max_length=max_length, truncation=True, padding="max_length")
    labels = tokenizer(batch["fr"], max_length=max_length, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(tokenisation, batched=True, remove_columns=["en", "fr"])

# ---------- Training ----------

training_args = Seq2SeqTrainingArguments(
    output_dir="./model",
    num_train_epochs=3,
    save_strategy="no",
    eval_strategy="epoch",
    logging_strategy="epoch",
    logging_steps=200,
    logging_dir="./logs",
    learning_rate=3e-5,
    lr_scheduler_type = "linear",
    warmup_ratio = 0.05,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    predict_with_generate=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["val"],
    tokenizer=tokenizer
)

trainer.train()

# ---------- Save final model for inference ----------
final_model_path = "./model/final"
trainer.save_model(final_model_path)   # Saves only what you need for inference
tokenizer.save_pretrained(final_model_path)  # Ensure tokenizer files are there
print(f"\nModel and tokenizer saved to {final_model_path}")