#!/usr/bin/env python3
"""
https://keras.io/examples/generative/text_generation_with_miniature_gpt/
"""
import time

import numpy as np
import tensorflow as tf

from typing import Optional

from base_model import PerformanceModel

from registry import register_param, register_links, register_link_parameter


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


@register_param('attn_heads', int,
                'Number of attention heads')
@register_param('embed_dim', int,
                'Dimension of output and \'value\' projection')
@register_param('attn_dim', Optional[int],
                'Dimension of \'key\' projection. Set to None to use embed_dim')
@register_param('sequence_length', int,
                'Maximum input sequence length')
class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Masked MultiHeadAttention as described in Vaswani et al. (2017)

    https://nn.labml.ai/transformers/mha.html

    Adjusted to allow for a query/key dimension (_d_k) that differs from the
        value and output dimensions (embed_dim).
    These are called (att) and (hs) respectively in Huang et al. (2018)
    """
    def __init__(self, **config):
        super().__init__()
        self._num_heads = config['attn_heads']
        self._embed_dim = config['embed_dim']
        self._attn_dim = config.get('attn_dim')
        self._max_seq_len = config['sequence_length']
        if self._attn_dim is None:
            self._attn_dim = self._embed_dim

        assert self._embed_dim % self._num_heads == 0, ('{num_heads} must be a divisor of {embed_dim}\n'
                                                        f'Got num_heads={self._num_heads} and embed_dim={self._embed_dim}')
        assert self._attn_dim % self._num_heads == 0, ('{num_heads} must be a divisor of {attn_dim}\n'
                                                       f'Got num_heads={self._num_heads} and attn_dim={self._attn_dim}')

        self._d_v = self._embed_dim // self._num_heads
        self._d_k = self._attn_dim // self._num_heads

        self.Q = tf.keras.layers.experimental.EinsumDense(
            'btd,dhk->bthk', output_shape=[self._max_seq_len, self._num_heads, self._d_k])
        self.K = tf.keras.layers.experimental.EinsumDense(
            'bsd,dhk->bshk', output_shape=[self._max_seq_len, self._num_heads, self._d_k])
        self.V = tf.keras.layers.experimental.EinsumDense(
            'bsd,dhv->bshv', output_shape=[self._max_seq_len, self._num_heads, self._d_v])

        self.scale = tf.cast(1.0 / tf.math.sqrt(float(self._d_k)), self.compute_dtype)

        self.softmax = tf.keras.layers.Softmax(axis=3)

        self.O = tf.keras.layers.experimental.EinsumDense(
            'bthv,dhv->btd', output_shape=[self._max_seq_len, self._embed_dim])

    def call(self, inputs, mask, training=False):
        # inputs: (B, S, d_model)
        q = self.Q(inputs, training=training)  # (B, T, h, d_k)
        k = self.K(inputs, training=training)  # (B, S, h, d_k)
        v = self.V(inputs, training=training)  # (B, S, h, d_v)

        if mask is not None:
            mask = tf.expand_dims(mask, axis=-3)  # Expand (B, T, S) to (B, h, T, S)

        attn_score = self.attention_scaled_dot_product(q, k)
        attn_score = self.softmax(attn_score, mask=mask, training=training)  # softmax along axis S

        x = tf.einsum('bhts,bshv->bthv', attn_score, v)  # (B, T, h, d_v)

        return self.O(x, training=training)  # (B, T, d_model)

    def attention_scaled_dot_product(self, q, k):
        # We scale the score before the multiplication by K.transpose because
        #   the element-wise mult is faster with fewer elements
        # So scaling here is faster when seq_len > d_k
        q *= self.scale
        attn_score = tf.einsum('bshk,bthk->bhts', k, q)  # (B, h, T, S)
        return attn_score


@register_param('max_relative_pos', int,
                'Clipping distance of relative positional encodings')
@register_links({'MultiHeadAttention'})
class RelativeGlobalAttention(MultiHeadAttention):
    """
    Huang et al. (2018)

    A variation of regular MultiHeadAttention, where information about distance
        between queries and keys is added to the dot-product attention.

    Distances further than max_relative_pos are clipped
    """
    def __init__(self, **config):
        super().__init__(**config)

        max_relative_pos = config.get('max_relative_pos')
        if max_relative_pos is None:
            self._max_relative_pos = self._max_seq_len
        else:
            self._max_relative_pos = max_relative_pos

        self.pos_emb = self.add_weight(name='positional_embedding_matrix',
                                       shape=(self._max_relative_pos, self._num_heads, self._d_k),
                                       trainable=True)

    def attention_scaled_dot_product(self, q, k):
        attn_score = tf.einsum('bshk,bthk->bhts', k, q)  # (B, h, T, S)

        assert q.shape[1] == k.shape[1]

        Er = self.clipped_relative_positions(q.shape[1])  # (S, h, d_k)
        QEr = tf.einsum('bthk,rhk->bhtr', q, Er)
        Srel = self.skew(QEr)  # (B, h, T, S)

        attn_score += Srel
        attn_score *= self.scale

        return attn_score

    def clipped_relative_positions(self, seq_len):
        """
        Calculate Er, clipping at a max distance of self._max_relative_pos
        """
        length_diff = seq_len - self._max_relative_pos

        # If the supplied sequence is larger than the relative position matrix (self.pos_emb),
        #     we repeat the embedding with the furthest distance
        if length_diff > 0:
            clip_pos = tf.expand_dims(self.pos_emb[0, :, :], axis=0)  # (1, h, d_k)
            clips = tf.tile(clip_pos, [length_diff, 1, 1])  # (num_clips, h, d_k)
            return tf.concat([clips, self.pos_emb], axis=0)  # (S, h, d_k)

        # Otherwise just truncate the embeddings if necessary
        start_pos = max(0, -length_diff)
        return self.pos_emb[start_pos:]

    @staticmethod
    def skew(QEr):
        # QEr: (B, h, seq_q, seq_r)
        x = tf.pad(QEr, tf.constant([[0, 0], [0, 0], [0, 0], [1, 0]]))
        x = tf.reshape(x, [-1, x.shape[1], x.shape[2]+1, x.shape[3]-1])
        return x[:, :, 1:, :]  # (B, h, seq_q, seq_r)


@register_link_parameter('attn_type', {
    'absolute': 'MultiHeadAttention',
    'relative': 'RelativeGlobalAttention'
})
@register_param('ff_dim', int,
                'Output dimension of the first dense sublayer')
@register_param('embed_dim', int,
                'Dimension of output and \'value\' projection')
@register_param('layernorm_eps', float,
                'Epsilon value used in LayerNorm sublayer')
@register_param('drop_rate', float,
                'Dropout rate to apply after attention and last dense sublayer')
class TransformerBlock(tf.keras.layers.Layer):
    """
    Transformer decoder layer consisting of one sublayer of (masked) MHA followed by two dense sublayers.
    Applies dropout and layer normalization as described in Vaswani et al. 2017.

    The sequence-to-sequence attention (using an encoder's output) is left out,
    since we are using the decoder only.
    """

    def __init__(self, name='transformer_block', **config):
        super().__init__(name=name)
        self.attn = self._attn_layer_from_config(**config)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(config['ff_dim'], activation='relu'),
            tf.keras.layers.Dense(config['embed_dim'])])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=config['layernorm_eps'])
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=config['layernorm_eps'])
        self.dropout1 = tf.keras.layers.Dropout(config['drop_rate'])
        self.dropout2 = tf.keras.layers.Dropout(config['drop_rate'])

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

    @staticmethod
    def _attn_layer_from_config(**config):
        if config['attn_type'] == 'absolute':
            return MultiHeadAttention(**config)
        if config['attn_type'] == 'relative':
            return RelativeGlobalAttention(**config)
        raise ValueError('Unsupported config.attn_type,'
                         'please select either \'absolute\' or \'relative\'.')


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

    def positional_encoding(self, position, d_model):
        angle_rads = PositionalEncoding.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=self.compute_dtype)


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


@register_param('sequence_length', int,
                'Maximum input sequence length')
@register_param('embed_dim', int,
                'Hidden dimension size')
@register_param('drop_rate', float,
                'Dropout rate to use after input layer and in TransformerBlock')
@register_param('num_layers', int,
                'Number of stacked TransformerBlock layers to use')
@register_links({'PerformanceModel', 'TransformerBlock'})
class TransformerModel(PerformanceModel):
    """
    Transformer decoder model based on Vaswani et al. 2017.
    Consists of an input embedder, positional encodings, transformer layers and output embedding.

    The input layer takes in a sequence and converts it into
        one-hot encoding before running it through the model,
        so you don't have to do that in some other place.
    The model outputs logits (the final softmax is omitted).

    The input and output embeddings use the same weights but transposed.
    This was described in the original Transformer paper but not present in every implementation.
    """

    def __init__(self,
                 model_name,
                 input_loader,
                 restore_checkpoint,
                 **config):
        super().__init__(model_name, input_loader, restore_checkpoint, **config)
        self._vocab_size = self.input_loader.vocab_size
        self._sequence_length = config['sequence_length']
        self._embed_dim = config['embed_dim']

        self.inp_emb = SharedTokenEmbedding(self._vocab_size, self._embed_dim)
        self.pos_enc = PositionalEncoding(self._sequence_length, self._embed_dim)
        self.inp_dropout = tf.keras.layers.Dropout(config['drop_rate'])

        self.transformer_stack = [
            TransformerBlock(name=f'transformer_block_{i}', **config)
            for i in range(config['num_layers'])
        ]
        self.out_emb = self.inp_emb  # last projection shares weights with input embedding
        # Softmax is omitted, model returns logits

    def call(self, inputs, training=False):
        # inputs: (batch, seq_len)
        x = tf.one_hot(inputs, self._vocab_size)
        # x: (batch, seq_len, vocab_size)
        x = self.inp_emb(x, encode=True, training=training)
        # x: (batch, seq_len, embed_dim)
        x *= tf.math.sqrt(tf.cast(self._embed_dim, self.compute_dtype))
        x += self.pos_enc(x, training=training)
        x = self.inp_dropout(x, training=training)

        for block in self.transformer_stack:
            x = block(x, training=training)

        return self.out_emb(x, encode=False, training=training)
        # output: (batch, seq_len, vocab_size)

    def generate_step(self, inputs, temperature):
        inputs_truncated = inputs[:, max(0, inputs.shape[1]-self._sequence_length):]
        predicted_logits = self(inputs_truncated, training=False)
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits *= temperature
        predicted_categories = tf.random.categorical(predicted_logits, num_samples=1, dtype=tf.int32)
        return predicted_categories

    def sample_music(self, sample_length=512, temperature=1.0, verbose=False):
        if self.input_loader.midi_processor.piece_start:
            primer = self.input_loader.midi_processor.start_token
        else:
            primer = 0
        result = tf.constant([[primer]]*1, shape=(1, 1), dtype=tf.int32)
        start = time.time()
        for i in range(sample_length):
            if verbose:
                print(f'Sampling... {i}/{sample_length}', end='\r')
            predicted_categories = self.generate_step(result, temperature)
            result = tf.concat([result, predicted_categories], axis=-1)
        if verbose:
            print(f'Sampled {sample_length} tokens in {time.time()-start:.2f} s.')
        return result
