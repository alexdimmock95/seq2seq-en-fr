from config import n_corpus_lines, vocab_size_bpe
import numpy as np
import re
from transformers import PreTrainedTokenizerFast
from keras.preprocessing.sequence import pad_sequences
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
  print('\nPreprocessing script running...\n')

# Import language dataset
data_path = "/Users/Alex/Documents/Coding/2. Data Scientist - Natural Language Processing Specialist/deep_learning_project/language_pair/fra-eng/fra.txt"

# Defining lines as a list of each line
with open(data_path, 'r', encoding='utf-8') as f:   # Open in read mode
  lines = f.read().split('\n')    # Split each line by newline

# Building empty lists to hold sentences
input_docs = []   # Initialise list for input docs (English)
target_docs = []    # Initialise list for target docs (French)

# Adjust number of lines to preprocess
for line in lines[:n_corpus_lines]:    # For each line in first N lines
  # Input and target sentences are separated by tabs
  input_doc, target_doc = line.split('\t')[:2]    # Splits each line at tab, takes index 0 and 1 and returns as input_doc and target_doc
  
  # Define function that tokenises text and concatenates with spaces into input and target docs
  def clean_and_add_tokens(text, is_input_doc=False):
    if is_input_doc:
      tokens = re.findall(r"[\w']+|[^\s\w]", text)
    else:
      tokens = re.findall(r"[\w]+|[^\s\w]", text)
    return " ".join(tokens)

  input_doc = clean_and_add_tokens(input_doc, is_input_doc=True)
  target_doc = clean_and_add_tokens(target_doc, is_input_doc=False)

  # Appending each input sentence to docs lists
  input_docs.append(input_doc)    # Add input doc to input docs list
  target_docs.append(target_doc)    # Add target doc to target docs list

# Split input and target docs into train_temp and test splits
input_docs_train_temp, input_docs_test, target_docs_train_temp, target_docs_test = train_test_split(input_docs, target_docs, test_size=0.2, random_state=42)

# Split train_temp into train and val splits
input_docs_train, input_docs_val, target_docs_train, target_docs_val = train_test_split(input_docs_train_temp, target_docs_train_temp, test_size=0.1, random_state=42)

# Combine cleaned input and target docs
corpus = input_docs + target_docs

# Create a blank BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()  # split on whitespace

# Define trainer with special tokens
trainer = BpeTrainer(
    vocab_size=vocab_size_bpe,
    min_frequency=2,
    special_tokens=["[PAD]","[UNK]", "[CLS]", "[SEP]", "[MASK]", "<START>", "<END>"]
)

# Train the tokenizer
tokenizer.train_from_iterator(corpus, trainer=trainer)

# Wrap it for HF
hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

# Calculate max sequence lengths from input and target docs
max_encoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", input_doc)) for input_doc in input_docs])
max_decoder_seq_length = max([len(re.findall(r"[\w]+|[^\s\w]", target_doc)) for target_doc in target_docs])

# Print max sequence lengths
if __name__ == "__main__":
  print(f'\nMax encoder seq length: {max_encoder_seq_length} \nMax decoder seq length: {max_decoder_seq_length} \n')

# Encode each sentence of input and target docs into BPE tokens
# Encode ALL English sentences to IDs
encoder_input_train = [hf_tokenizer.encode(sent, add_special_tokens=False) for sent in input_docs_train]
encoder_input_test  = [hf_tokenizer.encode(sent, add_special_tokens=False) for sent in input_docs_test]
encoder_input_val = [hf_tokenizer.encode(sent, add_special_tokens=False) for sent in input_docs_val]

# Create start and end IDs
start_id = hf_tokenizer.convert_tokens_to_ids("<START>")
end_id = hf_tokenizer.convert_tokens_to_ids("<END>")
hf_tokenizer.pad_token = "[PAD]"

# Encode ALL French sentences to IDs, then prepend/append START/END manually
decoder_input_train = [[start_id] + hf_tokenizer.encode(sent, add_special_tokens=False) for sent in target_docs_train]
decoder_input_test = [[start_id] + hf_tokenizer.encode(sent, add_special_tokens=False) for sent in target_docs_test]
decoder_input_val = [[start_id] + hf_tokenizer.encode(sent, add_special_tokens=False) for sent in target_docs_val]

decoder_target_train = [hf_tokenizer.encode(sent, add_special_tokens=False) + [end_id] for sent in target_docs_train]
decoder_target_test = [hf_tokenizer.encode(sent, add_special_tokens=False) + [end_id] for sent in target_docs_test]
decoder_target_val = [hf_tokenizer.encode(sent, add_special_tokens=False) + [end_id] for sent in target_docs_val]

# Pad all sequences to the same length
encoder_input_train = pad_sequences(encoder_input_train, padding='post')
encoder_input_test = pad_sequences(encoder_input_test, padding='post')
encoder_input_val = pad_sequences(encoder_input_val, padding='post')

decoder_input_train = pad_sequences(decoder_input_train, padding='post')
decoder_input_test = pad_sequences(decoder_input_test, padding='post')
decoder_input_val = pad_sequences(decoder_input_val, padding='post')

decoder_target_train = pad_sequences(decoder_target_train, padding='post')
decoder_target_test = pad_sequences(decoder_target_test, padding='post')
decoder_target_val = pad_sequences(decoder_target_val, padding='post')

# Print number of input and target tokens
def print_tokens():
  # Calculate total number of tokens in corpus
  num_corpus_tokens = hf_tokenizer.vocab_size

  def get_words(text, language='en'):
    if language == 'en':
        # keep contractions as one word
        return re.findall(r"[\w']+", text)
    else:
        # split French contractions into separate words
        return re.findall(r"[\w]+", text)

  # Input (English) vocab
  input_vocab = set()
  for sent in input_docs:
    input_vocab.update(get_words(sent, 'en'))

  # Target (French) vocab
  target_vocab = set()
  for sent in target_docs:
    target_vocab.update(get_words(sent, 'fr'))

  print(f'Corpus token size: {num_corpus_tokens}')
  print(f"English word count: {len(input_vocab)}")
  print(f"French word count: {len(target_vocab)}")

# Call the function to print tokens if this script is run (not imported as a module)
if __name__ == "__main__":
  print_tokens()

if __name__ == "__main__":
  print('\nPreprocessing script finished.\n')