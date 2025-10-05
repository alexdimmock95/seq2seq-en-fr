# ---------- Imports ----------
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------- Load Model ----------
model_path = "./model"  # same as output_dir above
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# ---------- Inference ----------
device = 'cpu'
model.to(device)

def translate(text, max_length=64):
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------- Translate ----------
if __name__ == "__main__":
    example_text = "This is an example sentence in English, now translated into French!"
    translation = translate(example_text)
    print(f"\nInput: {example_text}\nTranslation: {translation}\n")