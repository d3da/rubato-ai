#!/usr/bin/env python3
"""
https://keras.io/examples/generative/text_generation_with_miniature_gpt/
"""
from typing import Optional

import numpy as np
import tensorflow as tf


def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    1's in the lower triangle, counting from the lower right corner.
    """
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
    )
    return tf.tile(mask, mult)


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Vaswani et al. (2017)

    https://nn.labml.ai/transformers/mha.html

    Adjusted to allow for a query/key dimension (_d_k) that differs from the
        value and output dimensions (embed_dim).
    These are called (att) and (hs) respectively in Huang et al. (2018)
    """
    def __init__(self, num_heads, embed_dim, attn_dim):
        super().__init__()
        self._num_heads = num_heads
        self._embed_dim = embed_dim

        assert embed_dim % num_heads == 0, ('{num_heads} must be a divisor of {embed_dim}\n'
                                            f'Got num_heads={num_heads} and embed_dim={embed_dim}')
        assert attn_dim % num_heads == 0, ('{num_heads} must be a divisor of {attn_dim}\n'
                                           f'Got num_heads={num_heads} and attn_dim={attn_dim}')

        self._d_v = embed_dim // num_heads
        self._d_k = attn_dim // num_heads

        self.Q = tf.keras.layers.experimental.EinsumDense('bid,dhk->bihk', output_shape=[None, num_heads, self._d_k])
        self.K = tf.keras.layers.experimental.EinsumDense('bid,dhk->bihk', output_shape=[None, num_heads, self._d_k])
        self.V = tf.keras.layers.experimental.EinsumDense('bid,dhv->bihv', output_shape=[None, num_heads, self._d_v])

        self.scale = 1 / tf.math.sqrt(tf.cast(self._d_k, tf.float32))

        self.softmax = tf.keras.layers.Softmax(axis=1)

        self.O = tf.keras.layers.Dense(embed_dim)

    def call(self, inputs, mask, training=False):

        # inputs: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = inputs.shape

        q = self.Q(inputs, training=training)
        k = self.K(inputs, training=training)
        v = self.V(inputs, training=training)
        # q, k: (batch_size, seq_len, num_heads, d_k)
        # v: (batch_size, seq_len, num_heads, d_v)

        # We scale the score before the multiplication by K.transpose because
        #   the element-wise mult is faster with fewer elements
        # So scaling here is faster when seq_len > d_k
        q *= self.scale

        attn_score = tf.einsum('bihd,bjhd->bijh', q, k)  # Q x K.T
        # attn_score: (batch_size, seq_len_q, seq_len_k, num_heads)

        # mask: (batch_size, seq_len_q, seq_len_k)
        if mask is not None:
            mask = tf.expand_dims(mask, axis=-1)
        # mask: (batch_size, seq_len_q, seq_len_k, 1)

        attn_score = self.softmax(attn_score, mask=mask, training=training)  # softmax along seq_len_k
        # attn_score: (batch_size, seq_len_q, seq_len_k, num_heads)

        x = tf.einsum('bijh,bihv->bjhv', attn_score, v)  # multiplication by V
        # x: (batch_size, seq_len, num_heads, d_v)

        x = tf.reshape(x, (-1, seq_len, embed_dim))
        # x: (batch_size, seq_len, embed_dim), with embed_dim == d_v * num_heads

        return self.O(x, training=training)  # (batch_size, seq_len, embed_dim)


class TransformerBlock(tf.keras.layers.Layer):

    USE_CUSTOM_MHA = True

    def __init__(self, embed_dim, num_heads, ff_dim, drop_rate, attn_dim=None):
        super().__init__()
        if attn_dim is None:
            attn_dim = embed_dim
        if self.USE_CUSTOM_MHA:
            self.attn = MultiHeadAttention(num_heads, embed_dim, attn_dim)
        else:
            self.attn = tf.keras.layers.MultiHeadAttention(num_heads, embed_dim//num_heads, attn_dim//num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(embed_dim)])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, inputs, training=False):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)

        # inputs: (batch_size, seq_len, embed_dim)
        if self.USE_CUSTOM_MHA:
            attn_output = self.attn(inputs, causal_mask, training=training)
        else:
            attn_output = self.attn(inputs, inputs, attention_mask=causal_mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        attn_output = self.layernorm1(inputs + attn_output, training=training)
        ffn_output = self.ffn(attn_output, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(attn_output + ffn_output, training=training)


class InputEmbedding(tf.keras.layers.Layer):
    """
    Learned token embeddings are added to learned positional embeddings.

    from https://www.tensorflow.org/text/tutorials/transformer#encoder_and_decoder

    TODO split into distinct input embedding and positional encoding layers
    """
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.embed_dim = embed_dim
        self.pos_enc = InputEmbedding.positional_encoding(maxlen, embed_dim)

    def call(self, x, training=False):
        seq_len = tf.shape(x)[-1]
        # input embedding part
        x = self.token_emb(x, training=training)
        x *= tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))
        # positional encoding part
        return x + self.pos_enc[:, :seq_len, :]

    @staticmethod
    def get_angles(pos, i, d_model):
        # TODO tensorflow version
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    @staticmethod
    def positional_encoding(position, d_model):
        # TODO tensorflow version
        angle_rads = InputEmbedding.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)


class TransformerModel(tf.keras.layers.Layer):
    def __init__(self,
                 vocab_size: int,
                 sequence_length: int,
                 num_layers: int,
                 drop_rate: float,
                 embed_dim: int,
                 attn_heads: int,
                 ff_dim: int,
                 attn_dim: Optional[int]):
        super().__init__()
        self._sequence_length = sequence_length

        self.emb = InputEmbedding(sequence_length, vocab_size, embed_dim)
        self.transformer_stack = tf.keras.Sequential([
            TransformerBlock(embed_dim, attn_heads, ff_dim, drop_rate, attn_dim)
            for _ in range(num_layers)
        ])
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, training=False):
        # inputs: (B, L)
        x = self.emb(inputs, training=training)
        # x: (B, L, embed_dim)
        x = self.transformer_stack(x, training=training)
        # x: (B, L, embed_dim)
        return self.dense(x, training=training)

    def generate_step(self, inputs, temperature):
        inputs_truncated = inputs[:, max(0, inputs.shape[1]-self._sequence_length):]
        predicted_logits = self(inputs_truncated, training=False)
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits *= temperature
        predicted_categories = tf.random.categorical(predicted_logits, num_samples=1, dtype=tf.int32)
        return predicted_categories

    def sample_music(self, sample_length=8, temperature=1.0):
        primer = tf.constant([[0]]*1, shape=(1, 1), dtype=tf.int32)
        result = primer[:]
        for _ in range(sample_length):
            predicted_categories = self.generate_step(result, temperature)
            result = tf.concat([result, predicted_categories], axis=-1)
        return result
    
    def sample_bach_wtc1_no9(self, sample_length=1024, temperature=1.0):
        primer = tf.constant([[380, 258, 393, 64, 257, 390, 52, 305, 396, 68, 282, 192, 265, 398, 71, 294, 394, 56, 258, 396, 76, 260, 199, 259, 196, 286, 397, 75, 261, 204, 286, 398, 76, 263, 203, 288, 392, 57, 395, 75, 257, 204, 262, 203, 257, 396, 73, 262, 201, 257, 184, 261, 396, 75, 261, 399, 73, 262, 203, 263, 398, 71, 258, 201, 288, 398, 73, 273, 199, 275, 397, 76, 260, 201, 288, 393, 56, 264, 185, 281, 393, 54, 260, 184, 287, 396, 73, 204, 392, 56, 182, 261, 201, 258, 396, 71, 262, 199, 260, 398, 73, 263, 398, 71, 201, 263, 199, 259, 396, 69, 289, 397, 71, 259, 197, 292, 396, 76, 267, 199, 279, 391, 57, 257, 184, 293, 392, 56]],
                             shape=(1, 128), dtype=tf.int32)
        result = primer[:]
        for _ in range(sample_length):
            predicted_categories = self.generate_step(result, temperature)
            result = tf.concat([result, predicted_categories], axis=-1)
        return result

