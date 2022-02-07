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
    These are called (att) and (hs) respectively in Anna Huang et al. (2018)
    """
    def __init__(self, num_heads, embed_dim, attn_dim=None):
        super().__init__()
        self._num_heads = num_heads
        self._embed_dim = embed_dim

        assert embed_dim % num_heads == 0, ('{num_heads} must be a divisor of {embed_dim}\n'
                                            f'Got num_heads={num_heads} and embed_dim={embed_dim}')

        self._d_v = embed_dim // num_heads

        if attn_dim is None:
            self._d_k = embed_dim // num_heads
        else:
            self._d_k = attn_dim // num_heads
            assert attn_dim % num_heads == 0, ('{num_heads} must be a divisor of {attn_dim}\n'
                                                f'Got num_heads={num_heads} and attn_dim={attn_dim}')


        # self.Q = tf.keras.layers.Dense(embed_dim)
        # self.K = tf.keras.layers.Dense(embed_dim)
        # self.V = tf.keras.layers.Dense(embed_dim)

        self.Q = tf.keras.layers.experimental.EinsumDense('bid,dhk->bihk', output_shape=[None, num_heads, self._d_k])
        self.K = tf.keras.layers.experimental.EinsumDense('bid,dhk->bihk', output_shape=[None, num_heads, self._d_k])
        self.V = tf.keras.layers.experimental.EinsumDense('bid,dhv->bihv', output_shape=[None, num_heads, self._d_v])

        self.scale = 1 / tf.math.sqrt(tf.cast(self._d_k, tf.float32))

        self.softmax = tf.keras.layers.Softmax(axis=1)

        self.O = tf.keras.layers.Dense(embed_dim)

    def call(self, inputs, mask):

        # inputs: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = inputs.shape

        # q = tf.reshape(self.Q(inputs), (-1, seq_len, self._num_heads, self._d_k))
        # k = tf.reshape(self.Q(inputs), (-1, seq_len, self._num_heads, self._d_k))
        # v = tf.reshape(self.Q(inputs), (-1, seq_len, self._num_heads, self._d_k))
        # # q, k, v: (batch_size, seq_len, num_heads, d_k)

        q = self.Q(inputs)
        k = self.K(inputs)
        v = self.V(inputs)
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

        attn_score = self.softmax(attn_score, mask=mask)  # softmax along seq_len_k
        # attn_score: (batch_size, seq_len_q, seq_len_k, num_heads)

        x = tf.einsum('bijh,bihv->bjhv', attn_score, v)  # multiplication by V
        # x: (batch_size, seq_len, num_heads, d_v)

        x = tf.reshape(x, (-1, seq_len, embed_dim))
        # x: (batch_size, seq_len, embed_dim), with embed_dim == d_v * num_heads

        return self.O(x)  # (batch_size, seq_len, embed_dim)


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, drop_rate, attn_dim):
        super().__init__()
        # self.attn = tf.keras.layers.MultiHeadAttention(num_heads, embed_dim//num_heads, attn_dim//num_heads)
        self.attn = MultiHeadAttention(num_heads, embed_dim, attn_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(embed_dim)])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)

        # inputs: (batch_size, seq_len, embed_dim)
        # attn_output = self.attn(inputs, inputs, attention_mask=causal_mask)
        attn_output = self.attn(inputs, causal_mask)
        attn_output = self.dropout1(attn_output)
        attn_output = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(attn_output)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(attn_output + ffn_output)


class InputEmbedding(tf.keras.layers.Layer):
    """
    Learned token embeddings are added to learned positional embeddings.

    from https://www.tensorflow.org/text/tutorials/transformer#encoder_and_decoder
    """
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.embed_dim = embed_dim
        self.pos_enc = InputEmbedding.positional_encoding(maxlen, embed_dim)

    def call(self, x):
        seq_len = tf.shape(x)[-1]
        x = self.token_emb(x)
        return x + self.pos_enc[:, :seq_len, :]

    @staticmethod
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    @staticmethod
    def positional_encoding(position, d_model):
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

        self.emb = InputEmbedding(sequence_length, vocab_size, embed_dim)
        self.transformer_stack = tf.keras.Sequential([
            TransformerBlock(embed_dim, attn_heads, ff_dim, drop_rate, attn_dim)
            for _ in range(num_layers)
        ])
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, *args, **kwargs):
        x = self.emb(inputs)
        x = self.transformer_stack(x)
        return self.dense(x)

    @tf.function
    def generate_step(self, inputs, temperature):
        predicted_logits = self(inputs)
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits *= temperature
        predicted_categories = tf.random.categorical(predicted_logits, num_samples=1, dtype=tf.int32)
        return tf.concat([inputs, predicted_categories], axis=1)

    @tf.function
    def sample_music(self, start_event_category=0, sample_length=64, num_seqs=2, temperature=1.0):
        result = tf.constant([start_event_category]*num_seqs, shape=(num_seqs, 1), dtype=tf.int32)
        temperature = tf.constant(temperature, dtype=tf.float32)
        for _ in range(sample_length):
            result = self.generate_step(result, temperature)
        return result
