from config import latent_dim        # Paramaters defined in config.py
from layers import MaskedAttention
from preprocessing import hf_tokenizer, max_decoder_seq_length, input_docs_test, encoder_input_test, decoder_input_test, decoder_target_test

from tensorflow import keras
from keras.layers import Dot, Activation, Concatenate # To implement Luong attention
from keras.layers import Input, LSTM, Dense 
from keras.models import load_model, Model
import numpy as np

# ATTENTION DEBUGGING
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention(attention_weights, input_tokens, output_tokens):
    """
    attention_weights: (dec_timesteps, enc_timesteps)
    input_tokens: list of source tokens (strings)
    output_tokens: list of predicted tokens (strings)
    """
    plt.figure(figsize=(10,8))
    sns.heatmap(attention_weights, xticklabels=input_tokens, yticklabels=output_tokens, cmap='viridis')
    plt.xlabel('Encoder Input Tokens')
    plt.ylabel('Decoder Output Tokens')
    plt.title('Attention Heatmap')
    plt.show()

if __name__ == "__main__":
  print('\nTest script running...\n')

# Load model from training file
training_model = keras.models.load_model('training_model.keras', custom_objects={"MaskedAttention": MaskedAttention})

# ---------- Encoder model for inference ----------

# Extract encoder input layer from trained model
encoder_inputs = training_model.input[0]

# Extract encoder embedding layer from trained model
encoder_embedding_layer = training_model.get_layer("encoder_embedding")

# Pass encoder_inputs through encoder_embedding to get dense vector for each timestep
embedded_encoder_inputs = encoder_embedding_layer(encoder_inputs)

# Extract the encoder LSTM layer from trained model
encoder_lstm = training_model.get_layer("encoder_lstm")

# Save progressive results and final states for LSTM on embedded encoder inputs
encoder_outputs, state_h_enc, state_c_enc = encoder_lstm(embedded_encoder_inputs)

# Save hidden and cell states in a list
encoder_states = [state_h_enc, state_c_enc]

# Create inference encoder model
# Inputs:
#   encoder_inputs: source sequence (token IDs), shape (batch_size, encoder_timesteps)
# Outputs:
#   state_h_enc: encoder's final hidden state, shape (batch_size, latent_dim)
#   state_c_enc: encoder's final cell state, shape (batch_size, latent_dim)
encoder_model = Model(encoder_inputs, [encoder_outputs, state_h_enc, state_c_enc])

# ---------- Decoder model for inference ----------

# Extract decoder model from trained model:
decoder_inputs = Input(shape=(1,), dtype='int32', name='dec_token_in') # shape(1,) ensures passing one token at a time

# Extract embedding layer from trained model
decoder_embedding_layer = training_model.get_layer("decoder_embedding") # Embedding weights used during training are used here

# Extract the decoder LSTM from trained model
decoder_lstm = training_model.get_layer("decoder_lstm") # LSTM is from training model

# Extract the decoder Dense layer from trained model
decoder_dense = training_model.get_layer("decoder_dense") # Dense probability calcs from training model

# New inputs for inference
  # decoder_state_input_hidden: placeholder for decoder's previous hidden state (h) at current timestep
  # decoder_state_input_cell: placeholder for decoder's previous cell state (c) at current timestep
  # decoder_states_inputs: list combining hidden and cell state inputs
  # encoder_outputs_input: input placeholder for the encoder’s sequence of hidden states 
decoder_state_input_hidden = Input(shape=(latent_dim,), name='decoder_h_in')
decoder_state_input_cell = Input(shape=(latent_dim,), name='decoder_c_in')
decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]
encoder_outputs_input = Input(shape=(None, latent_dim), name="encoder_outputs_input")

# Pass IDs through embedding
embedded_decoder_inputs = decoder_embedding_layer(decoder_inputs) # Turn integer token ID into dense vector 

# Pass through LSTM
decoder_outputs, state_hidden, state_cell = decoder_lstm(embedded_decoder_inputs, initial_state=decoder_states_inputs) # Feeds in one token and last states, returns next softmax distribution, updated states

# Update states
decoder_states = [state_hidden, state_cell] # These will be carried forward into the next timestep

# ---------- Attention ----------

# Dot product between decoder hidden states and encoder outputs. 3rd axis (index 2, latent_dim) of both outputs into dot prod. Final shape is (batch_size, dec_timesteps, enc_timesteps)
dec_attention_scores = Dot(axes=[2, 2], name="dec_attention_scores")([decoder_outputs, encoder_outputs_input])

# Normalises attention scores so that each decoder timestep "weights" the encoder tokens appropriately
dec_attention_weights = Activation('softmax', name="dec_attention_weights")(dec_attention_scores)

# Context vector: weighted sum of encoder outputs. Final shape is (batch_size, dec_timesteps, latent_dim).
dec_context_vector = Dot(axes=[2,1], name="dec_context_vector")([dec_attention_weights, encoder_outputs_input])

# Concatenate context with decoder outputs. context_vector axis -1 gets added to decoder_outputs axis -1. Length of axis -1 is now 2*latent_dim. Final shape is (batch_size, dec_timesteps, 2*latent_dim)
decoder_combined_context = Concatenate(axis=-1, name="decoder_context")([dec_context_vector, decoder_outputs])

# Final dense
decoder_outputs = decoder_dense(decoder_combined_context) # Converts LSTM output into vocab probabilities

# Inference decoder model
decoder_model = Model(
  [decoder_inputs] + decoder_states_inputs + [encoder_outputs_input], # Input: current token + decoder_states_inputs
  [decoder_outputs, dec_attention_weights] + decoder_states # Output: softmax over vocab + decoder states
)

def decode_sequence(test_input):
  '''
  Description:
  Arguments: test_input: sequence from input dataset
  Outputs: decoded_sentence: predicted output sequence in target dataset
  '''
  
  # Encode the input as state h and c vectors. Result of calling .predict() will give final hidden and cell states for that input.
  encoder_outs, states_value_h, states_value_c = encoder_model.predict(test_input)

  # Save final hidden and cell states at states_value
  states_value = [states_value_h, states_value_c]

  # Find index of START and END tokens
  start_token_index = hf_tokenizer.convert_tokens_to_ids("<START>")
  end_token_index = hf_tokenizer.convert_tokens_to_ids("<END>")

  # Generate empty target sequence of length 1.
  target_seq = np.array([[start_token_index]])

  # Sampling loop for a batch of sequences
  # (to simplify, here we assume a batch of size 1)
  decoded_sentence = ''

  decoded_token_ids = []

  # ATTENTION DEBUGGING
  attention_matrix = []  # collect dec_attention_weights each timestep

  stop_condition = False

  while not stop_condition:
    
    # Run the decoder model
    # Inputs:
      # [target_seq]:the last token
      # states_value: the last hidden and cell states
    # Outputs: 
      # output_tokens: softmax probabilities for target token
      # hidden_state: short-term memory saved from last token
      # cell_state: long-term memory saved from last token

    output_tokens, dec_attn_weights, hidden_state, cell_state = decoder_model.predict(
      [target_seq] + states_value + [encoder_outs])
    
    # ATTENTION DEBUGGING
    attention_matrix.append(dec_attn_weights[0,0,:])  # assuming batch_size=1, dec timestep=1

    # Choose token with highest probability
    sampled_token_index = np.argmax(output_tokens[0, -1, :]) # Greedy encoding grabbing the highest probability from the vocab list dimension at first index and last timestep

    # Exit condition: either find stop token or hit max length
    if (sampled_token_index == end_token_index or len(decoded_token_ids) > max_decoder_seq_length):
      stop_condition = True
    else:
      # Append token ID to list
      decoded_token_ids.append(sampled_token_index)

    # Update the target sequence (of length 1) with the latest token index
    target_seq = np.array([[sampled_token_index]])

    # Update states
    states_value = [hidden_state, cell_state]
  
  '''  
  # After decoding
  attention_matrix = np.array(attention_matrix)  # shape: (dec_timesteps, enc_timesteps)
  input_labels = hf_tokenizer.convert_ids_to_tokens(test_input[0])
  decoder_labels = hf_tokenizer.convert_ids_to_tokens(decoded_token_ids)
  plot_attention(attention_matrix, input_labels, decoder_labels)
  '''
  # Decode all tokens at once
  decoded_sentence = hf_tokenizer.decode(decoded_token_ids, skip_special_tokens=True)

  return decoded_sentence

#### DEBUGGING ##################################

# Example: pick first 5 sentences
sample_indices = [0, 1, 2, 3, 4]

for i in sample_indices:
    input_text = input_docs_test[i]   # source sentence (English)
    target_text = hf_tokenizer.decode(decoder_target_test[i], skip_special_tokens=True)  # target sentence (French)
    
    input_ids = encoder_input_test[i]
    target_ids = decoder_target_test[i]
    
    print(f"\n=== Sample {i} ===")
    print("Input text: ", input_text)
    print("Input IDs : ", input_ids)
    print("Target text:", target_text)
    print("Target IDs:", target_ids)

    decoded_input_tokens = hf_tokenizer.convert_ids_to_tokens(input_ids)
    print("Decoded input tokens:", decoded_input_tokens)

    decoded_target_tokens = hf_tokenizer.convert_ids_to_tokens(target_ids)
    print("Decoded target tokens:", decoded_target_tokens)

    decoded_sentence = decode_sequence(encoder_input_test[i:i+1])
    print("Decoded sentence:", decoded_sentence)

#### DEBUGGING ##################################

# ---------- Translate ----------

n = 5  # number of random sentences to translate
num_samples = encoder_input_test.shape[0]

indices = np.arange(num_samples)
np.random.shuffle(indices)

for seq_index in indices[:n]:
    test_input = encoder_input_test[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(test_input)
    print('Input sentence:', input_docs_test[seq_index])
    print('Decoded sentence:', decoded_sentence)

if __name__ == "__main__":
  print('\nTest script finished.\n')

'''
Next steps:
- Bleu Score evaluation in training file???
'''