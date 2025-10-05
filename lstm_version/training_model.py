from config import latent_dim, embedding_dim, batch_size, epochs        # Paramaters defined in config.py
from layers import MaskedAttention
from preprocessing import hf_tokenizer, encoder_input_train, decoder_input_train, decoder_target_train, encoder_input_val, decoder_input_val, decoder_target_val

import matplotlib.pyplot as plt

# Add Dense to the imported layers
from keras.layers import Input, LSTM, Dense, Embedding, Lambda, Layer
from keras.layers import Dot, Activation, Concatenate # To implement Luong attention
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


if __name__ == "__main__":
  print('\nTraining script running...\n')

def build_training_model(num_tokens, latent_dim, embedding_dim):
    """
    Builds and compiles a seq2seq model for training.
    
    Args:
        num_tokens (int): Number of tokens for input + target corpus.
        latent_dim (int): Dimensionality of LSTM hidden states.
        embedding_dim (int): Dimensionality of the embedding layers.
    
    Returns:
        model (Model): Compiled seq2seq training model.
        encoder_lstm (LSTM): Encoder LSTM layer (needed for inference if you rebuild encoder_model).
        decoder_lstm (LSTM): Decoder LSTM layer (needed for inference).
        decoder_dense (Dense): Dense output layer for the decoder.
    """

    # ---------- Encoder ----------
    """ 
    Runs embedded encoder inputs through LSTM, producing final hidden and cell states.
    """
    # Keras input tensor placeholder expecting token indices of shape (batch_size, None (timestep)).
    encoder_inputs = Input(shape=(None, ), dtype='int32', name='encoder_input')
    
    # Produces an Embedding layer instance of shape (batch_size, timesteps, embedding_dim), which converts encoder_input into a dense vector of len(embedding_dim). Token 0 is ignored because it is padding.
    encoder_embedding = Embedding(input_dim=num_tokens, 
                                  output_dim=embedding_dim, 
                                  mask_zero=True,
                                  name='encoder_embedding')(encoder_inputs)
    
    # Creates an LSTM layer with shape (batch_size, latent_dim)
    encoder_lstm = LSTM(latent_dim, dropout=0.2, recurrent_dropout=0, return_sequences=True, return_state=True, name='encoder_lstm')

    # Calls the LSTM on the embedded encoder inputs. Encoder_outputs saved for attention inclusion.
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)

    # Packs the two states into a list
    encoder_states = [state_h, state_c]

    # ---------- Decoder ----------
    """ 
    Runs embedded decoder inputs through LSTM, producing step-by-step hidden and cell states.
    Calculates softmax probability on each output for the target vocabulary.
    """
    # Keras input tensor placeholder expecting token indices of shape (batch_size, None (timestep)).
    decoder_inputs = Input(shape=(None, ), name='decoder_input')
    
    # Produces an Embedding layer instance of shape (batch_size, timesteps, embedding_dim), which converts decoder_input into a dense vector of len(embedding_dim). Token 0 is ignored because it is padding.
    decoder_embedding = Embedding(input_dim=num_tokens, 
                                  output_dim=embedding_dim, 
                                  mask_zero=True, 
                                  name='decoder_embedding')(decoder_inputs)
    
    # Creates an LSTM layer with shape (batch_size, latent_dim)    
    decoder_lstm = LSTM(latent_dim, dropout=0.2, recurrent_dropout=0, return_sequences=True, return_state=True, name='decoder_lstm')
    
    # Calls the LSTM on the embedded decoder inputs. state_h and state_c ignored. Decoder outputs needed to feed into Dense layer.
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

    # ---------- Attention ----------

    # Dot product between decoder hidden states and encoder outputs. 3rd axis (index 2, latent_dim) of both outputs into dot prod. Final shape is (batch_size, dec_timesteps, enc_timesteps)
    attention_scores = Dot(axes=[2, 2], name="attention_scores")([decoder_outputs, encoder_outputs])

    # Mask from embedding layer
    encoder_mask = encoder_embedding._keras_mask  # shape: (batch_size, enc_timesteps)

    # Set masked positions to -inf before softmax
    attention_scores_masked = MaskedAttention(name="masked_attention")([encoder_mask, attention_scores])

    # Normalises attention scores so that each decoder timestep "weights" the encoder tokens appropriately
    attention_weights = Activation('softmax', name="attention_weights")(attention_scores_masked)

    # Context vector: weighted sum of encoder outputs. Final shape is (batch_size, dec_timesteps, latent_dim).
    context_vector = Dot(axes=[2,1], name="context_vector")([attention_weights, encoder_outputs])

    # Concatenate context with decoder outputs. context_vector axis -1 gets added to decoder_outputs axis -1. Length of axis -1 is now 2*latent_dim. Final shape is (batch_size, dec_timesteps, 2*latent_dim)
    decoder_combined_context = Concatenate(axis=-1, name="decoder_context")([context_vector, decoder_outputs])

    # ---------- Dense ----------

    # Converts raw scores into probabilities adding to 1 (softmax)
    decoder_dense = Dense(num_tokens, activation='softmax', name='decoder_dense')
    
    # Convert decoder_outputs (shape (batch_size, timesteps, latent_dim)) to a vector of length num_decoder_tokens
    decoder_outputs = decoder_dense(decoder_combined_context)

    # ---------- Full Seq2Seq Model ----------
    
    # encoder_inputs: batch of source sequences, shape (batch_size, encoder_timesteps), integers
    # decoder_inputs: batch of target sequences, shape (batch_size, decoder_timesteps), integers (teacher forcing, starts with <start>)
    # decoder_outputs: batch of predicted token probabilities, shape (batch_size, decoder_timesteps, num_decoder_tokens)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer=Adam(learning_rate=0.001, clipnorm=1.0), loss='sparse_categorical_crossentropy', metrics=['accuracy']) #Sparse_categorical_crossentropy as loss when target sequences are integers

    return model, encoder_lstm, decoder_lstm, decoder_dense

num_tokens = len(hf_tokenizer.get_vocab())

# Build model
training_model, encoder_lstm, decoder_lstm, decoder_dense = build_training_model(num_tokens, latent_dim, embedding_dim)

# Train model
if __name__ == "__main__":
    print(f'\nTraining the model:\n')

# Stop training if val_loss doesn't improve for 5 consecutive epochs
early_stop = EarlyStopping(
    monitor='val_loss',   # metric to monitor
    patience=5,           # number of epochs with no improvement before stopping
    restore_best_weights=True  # after stopping, restore weights from best epoch
)

# Reduce learning rate on plateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=1e-5
)

# ---------- Fit training model ----------
# Inputs:
#   encoder_input_data: source sequences, shape (num_samples, encoder_timesteps)
#   decoder_input_data: target sequences shifted right (teacher forcing), shape(num_samples, decoder_timesteps)
# Targets:
#   decoder_target_data: true next tokens, shape (num_samples, decoder_timesteps)
# batch_size: number of sequences per batch
# epochs: number of full dataset passes
# validation_split: fraction of data used for validation
history = training_model.fit(
    [encoder_input_train, decoder_input_train],
    decoder_target_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data = ([encoder_input_val, decoder_input_val], decoder_target_val),
    callbacks=[early_stop, reduce_lr]
)

# Print model summary
if __name__ == "__main__":
    print('\nModel summary:\n')
training_model.summary()

# Save model
training_model.save('training_model.keras')
print('\nModel saved to training_model.keras')

'''
# Plot training history values
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
'''

if __name__ == "__main__":
  print('\nTraining script finished.\n')