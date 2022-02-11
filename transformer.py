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
    def __init__(self, num_heads, embed_dim, attn_dim, max_seq_len):
        super().__init__()
        self._num_heads = num_heads
        self._embed_dim = embed_dim
        self._max_seq_len = max_seq_len

        assert embed_dim % num_heads == 0, ('{num_heads} must be a divisor of {embed_dim}\n'
                                            f'Got num_heads={num_heads} and embed_dim={embed_dim}')
        assert attn_dim % num_heads == 0, ('{num_heads} must be a divisor of {attn_dim}\n'
                                           f'Got num_heads={num_heads} and attn_dim={attn_dim}')

        self._d_v = embed_dim // num_heads
        self._d_k = attn_dim // num_heads

        self.Q = tf.keras.layers.experimental.EinsumDense('btd,dhk->bthk', output_shape=[max_seq_len, num_heads, self._d_k])
        self.K = tf.keras.layers.experimental.EinsumDense('bsd,dhk->bshk', output_shape=[max_seq_len, num_heads, self._d_k])
        self.V = tf.keras.layers.experimental.EinsumDense('bsd,dhv->bshv', output_shape=[max_seq_len, num_heads, self._d_v])

        self.scale = 1.0 / tf.math.sqrt(float(self._d_k))

        self.softmax = tf.keras.layers.Softmax(axis=3)

        self.O = tf.keras.layers.experimental.EinsumDense('bthv,dhv->btd', output_shape=[max_seq_len, embed_dim])

    def call(self, inputs, mask, training=False):
        # inputs: (B, S, d_model)
        q = self.Q(inputs, training=training)  # (B, T, h, d_k)
        k = self.K(inputs, training=training)  # (B, S, h, d_k)
        v = self.V(inputs, training=training)  # (B, S, h, d_v)

        # We scale the score before the multiplication by K.transpose because
        #   the element-wise mult is faster with fewer elements
        # So scaling here is faster when seq_len > d_k
        q *= self.scale

        attn_score = tf.einsum('bshk,bthk->bhts', k, q)  # (B, h, T, S)

        if mask is not None:
            mask = tf.expand_dims(mask, axis=-3)  # Expand (B, T, S) to (B, h, T, S)

        attn_score = self.softmax(attn_score, mask=mask, training=training)  # softmax along axis S

        x = tf.einsum('bhts,bshv->bthv', attn_score, v)  # (B, T, h, d_v)

        return self.O(x, training=training)  # (B, T, d_model)


class TransformerBlock(tf.keras.layers.Layer):

    def __init__(self, embed_dim, num_heads, ff_dim, drop_rate, sequence_length, attn_dim=None):
        super().__init__()
        if attn_dim is None:
            attn_dim = embed_dim
        self.attn = MultiHeadAttention(num_heads, embed_dim, attn_dim, sequence_length)
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
        attn_output = self.attn(inputs, causal_mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        attn_output = self.layernorm1(inputs + attn_output, training=training)
        ffn_output = self.ffn(attn_output, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(attn_output + ffn_output, training=training)


class PositionalEncoding(tf.keras.layers.Layer):
    """
    Learned token embeddings are added to learned positional embeddings.

    from https://www.tensorflow.org/text/tutorials/transformer#encoder_and_decoder

    TODO learned/linear embeddings with hparams
    """
    def __init__(self, max_seq_len, embed_dim):
        super().__init__()
        self.pos_enc = self.positional_encoding(max_seq_len, embed_dim)

    def call(self, x, training=False):
        return self.pos_enc[:, :x.shape[1], :x.shape[2]]

    @staticmethod
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    @staticmethod
    def positional_encoding(position, d_model):
        angle_rads = PositionalEncoding.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)


class SharedTokenEmbedding(tf.keras.layers.Layer):
    """
    Section 3.4 of Vaswani et al. (2017):
        The input embedding matrix and the linear projection
        before the final softmax in the decoder share weights,
        similar to Press & Wolf (2016)
    """
    def __init__(self, token_dim, model_dim):
        super().__init__()
        self.emb_matrix = self.add_weight(name='shared_embedding', shape=(token_dim, model_dim), trainable=True)

    def call(self, inputs, encode=True, training=None):
        if encode:
            # inputs: (batch, sequence, token_dim)
            return tf.einsum('bst,tm->bsm', inputs, self.emb_matrix)

        # inputs: (batch, sequence, model_dim)
        return tf.einsum('bsm,tm->bst', inputs, self.emb_matrix)


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
        self._vocab_size = vocab_size
        self._sequence_length = sequence_length
        self._embed_dim = embed_dim

        self.inp_emb = SharedTokenEmbedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoding(sequence_length, embed_dim)
        self.inp_dropout = tf.keras.layers.Dropout(drop_rate)

        self.transformer_stack = tf.keras.Sequential([
            TransformerBlock(embed_dim, attn_heads, ff_dim, drop_rate, sequence_length, attn_dim)
            for _ in range(num_layers)
        ])
        self.out_emb = self.inp_emb  # last projection shares weights with input embedding
        # Softmax is omitted, model returns logits

    def call(self, inputs, training=False):
        # inputs: (batch, seq_len)
        x = tf.one_hot(inputs, self._vocab_size)
        # x: (batch, seq_len, vocab_size)
        x = self.inp_emb(x, encode=True, training=training)
        # x: (batch, seq_len, embed_dim)
        x *= tf.math.sqrt(tf.cast(self._embed_dim, tf.float32))
        x += self.pos_enc(x, training=training)

        x = self.transformer_stack(x, training=training)
        return self.out_emb(x, encode=False, training=training)
        # output: (batch, seq_len, vocab_size)

    def generate_step(self, inputs, temperature):
        inputs_truncated = inputs[:, max(0, inputs.shape[1]-self._sequence_length):]
        predicted_logits = self(inputs_truncated, training=False)
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits *= temperature
        predicted_categories = tf.random.categorical(predicted_logits, num_samples=1, dtype=tf.int32)
        return predicted_categories

    def sample_music(self, sample_length=512, temperature=1.0):
        primer = tf.constant([[0]]*1, shape=(1, 1), dtype=tf.int32)
        result = primer[:]
        for _ in range(sample_length):
            predicted_categories = self.generate_step(result, temperature)
            result = tf.concat([result, predicted_categories], axis=-1)
        return result
