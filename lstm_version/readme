# English-French-Neural-Machine-Translation-with-Seq2Seq-GRU-LSTM-
English–French Neural Machine Translation

This project explores neural machine translation (NMT) by building a sequence-to-sequence (seq2seq) encoder–decoder model to translate English into French.

# Objective

The aim of the project is to train a model that can learn bilingual mappings from parallel English–French text data and generate translations at the sentence level.

# Approach
	•	Implemented an encoder–decoder architecture using recurrent neural networks (starting with LSTM, later tested with GRU for efficiency).
	•	Used embedding layers to represent integer token indices as dense vectors, instead of relying on large one-hot encodings.
	•	Trained initially on a subset of ~10–20k sentence pairs to validate the method before scaling to the full dataset.
	•	Experimented with different optimisers (Adam vs RMSprop) and tuning hyperparameters.

# Key Learnings
	•	Encoder–decoder RNNs can begin producing recognisable translations after around 10 epochs, with fluency improving as dataset size and training time increase.
	•	GRUs provide a lighter, faster alternative to LSTMs while often achieving comparable quality.
	•	Embedding-based inputs are far more memory-efficient than one-hot representations.
	•	Training loss decreased from 1.6 to 1.0, showing the model’s capacity to capture cross-lingual structure.

# Next Steps
	•	Add attention mechanisms (Bahdanau or Luong attention) to improve translation quality, especially for longer sequences.
	•	Compare recurrent approaches with Transformer-based architectures (for example, via KerasNLP).
	•	Expand the dataset and refine tokenisation, exploring subword approaches such as Byte Pair Encoding.
