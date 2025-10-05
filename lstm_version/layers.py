import tensorflow as tf
from keras.layers import Layer

# Define class to create attention masks
class MaskedAttention(Layer):
    def __init__(self, **kwargs):
        super(MaskedAttention, self).__init__(**kwargs)

    def call(self, inputs):
        encoder_mask, attention_scores = inputs
        # Expand mask: (batch_size, 1, enc_timesteps)
        mask_expanded = tf.expand_dims(encoder_mask, axis=1)
        # Apply mask: keep scores where mask==True, else set to -1e9
        masked_scores = tf.where(mask_expanded, attention_scores, tf.fill(tf.shape(attention_scores), -1e9))
        return masked_scores

    def get_config(self):
        config = super(MaskedAttention, self).get_config()
        return config