import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Permute, Multiply
import tensorflow.keras.backend as K

class CustomAttention(Layer):
    """
    Custom Attention layer for sequence classification using Additive Attention.
    (Corrected to define internal layers only once in __init__)
    """
    def __init__(self, **kwargs):
        super(CustomAttention, self).__init__(**kwargs)
        
        # 1. Define layers here (in __init__), NOT in call()
        self.dense_score = Dense(1, activation='tanh', name='attention_score')
        
        # We need to define the Permute and Multiply layers as attributes
        # Permute layers generally don't have weights, but defining them
        # ensures they are part of the graph defined at build time.
        self.permute_scores = Permute((2, 1), name='permute_scores')
        self.permute_weights = Permute((2, 1), name='permute_weights')
        self.multiply = Multiply(name='weighted_sequence')
        
        # Note: We need a Dense layer for the softmax activation
        # This layer's output dimension depends on the sequence length, 
        # which can be tricky. A simpler and safer approach is to use 
        # the K.softmax function directly, which doesn't create new variables.

    def build(self, input_shape):
        # We don't need the Dense layer for softmax, so we can define it in build
        # if necessary, but here we can rely on K.softmax in call().
        super(CustomAttention, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, seq_length, n_features)

        # 1. Calculate Alignment Scores (e_i) - Uses layer defined in __init__
        # scores shape: (batch_size, seq_length, 1)
        scores = self.dense_score(inputs)

        # 2. Normalize Scores to get Attention Weights (alpha_i)
        # Permute from (batch, seq_len, 1) to (batch, 1, seq_len)
        attention_weights = self.permute_scores(scores)
        
        # Use K.softmax directly on the required axis (last axis is sequence length)
        # to avoid defining a Dense layer here which would create variables.
        attention_weights = K.softmax(attention_weights, axis=-1)
        
        # Permute back to (batch, seq_len, 1)
        attention_weights = self.permute_weights(attention_weights)
        
        # 3. Create the Context Vector (c)
        # Multiply the input sequence (H) by the learned weights (alpha_i)
        weighted_sequence = self.multiply([inputs, attention_weights])
        
        return weighted_sequence