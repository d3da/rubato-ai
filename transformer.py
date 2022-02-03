#!/usr/bin/env python3
"""
https://keras.io/examples/generative/text_generation_with_miniature_gpt/
"""
# import os
# import pdb
# import sys
# import time

import tensorflow as tf

# from input_loader import PerformanceInputLoader
# from performance_rnn import PROJECT_DIR, TrainCallback


# PROJECT_DIR = os.path.dirname(__file__)


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


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, drop_rate=0.1):
        super().__init__()
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads, embed_dim)
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
        attn_output = self.attn(inputs, inputs, attention_mask=causal_mask)
        attn_output = self.dropout1(attn_output)
        attn_output = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(attn_output)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(attn_output + ffn_output)


class InputEmbedding(tf.keras.layers.Layer):
    """
    Learned token embeddings are added to learned positional embeddings.
    """
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


class TransformerModel(tf.keras.Model):
    def __init__(self,
                 # input_loader,
                 # model_name, train_dir,
                 # restore_checkpoint,
                 vocab_size,
                 sequence_length,
                 num_layers = 2, drop_rate = 0.1,
                 embed_dim = 128, attn_heads = 2,
                 ff_dim = 1024):
                # learning_rate):
        super().__init__(self)
        # super().__init__(self, name=model_name)
        # self.input_loader = input_loader
        # vocab_size = input_loader.vocab_size

        self.emb = InputEmbedding(sequence_length, vocab_size, embed_dim)
        self.transformer_blocks = [
            TransformerBlock(embed_dim, attn_heads, ff_dim, drop_rate)
            for _ in range(num_layers)
        ]
        self.dense = tf.keras.layers.Dense(vocab_size)

        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # self.compile(optimizer=self.optimizer, loss=self.loss,
        #              metrics=['accuracy'])
        #
        # # Setup checkpoints
        # checkpoint_dir = os.path.join(train_dir, 'checkpoints', model_name)
        # checkpoint = tf.train.Checkpoint(model=self, optimizer=self.optimizer)
        # self.batch_ctr = tf.Variable(0, trainable=False, dtype=tf.int64)
        # self.epoch_ctr = tf.Variable(0, trainable=False, dtype=tf.int64)
        # self.chkpt_mgr = tf.train.CheckpointManager(
        #     checkpoint,
        #     directory=checkpoint_dir,
        #     max_to_keep=50,
        #     step_counter=self.batch_ctr)
        # if restore_checkpoint:
        #     checkpoint.restore(self.chkpt_mgr.latest_checkpoint)
        #     if self.batch_ctr.value() != 0:
        #         print(f'Restored checkpoint (batch {self.batch_ctr.value()}, epoch {self.epoch_ctr.value()})')
        #     else:
        #         print('Initialized model')
        #
        # self.callbacks = [TrainCallback(train_dir=train_dir)]
        # self.load_time = time.localtime()

    def call(self, inputs, *args, **kwargs):
        x = self.emb(inputs)
        for block in self.transformer_blocks:
            x = block(x)
        return self.dense(x)

    # def train(self, epochs):
    #     return self.fit(self.input_loader.dataset, epochs=epochs, callbacks=self.callbacks)

    @tf.function
    def generate_step(self, inputs, temperature):
        predicted_logits = self(inputs)
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits *= temperature
        predicted_categories = tf.random.categorical(predicted_logits, num_samples=1, dtype=tf.int32)
        return tf.concat([inputs, predicted_categories], axis=1)

    @tf.function
    def sample_music(self, start_event_category=0, sample_length=512, num_seqs=2, temperature=1.0):
        result = tf.constant([start_event_category]*num_seqs, shape=(num_seqs, 1), dtype=tf.int32)
        temperature = tf.constant(temperature, dtype=tf.float32)
        for _ in range(sample_length):
            result = self.generate_step(result, temperature)
        return result


# if __name__ == '__main__':
#
#     dataset_base = os.path.join(PROJECT_DIR, 'data/maestro-v3.0.0')
#     dataset_csv = os.path.join(dataset_base, 'maestro-v3.0.0.csv')
#
#     input_loader = PerformanceInputLoader(
#         dataset_base,
#         dataset_csv,
#         sequence_length=512,
#         min_stride=128,
#         max_stride=256,
#         batch_size=64,
#         augmentation='aug-'
#     )
#
#     model = TransformerModel(
#         input_loader=input_loader,
#         model_name='base_transformer',
#         train_dir=PROJECT_DIR,
#         restore_checkpoint=True,
#         num_layers=6,
#         drop_rate=0.1,
#         embed_dim=128,
#         attn_heads=4,
#         ff_dim=128,
#         seq_len=512,
#         learning_rate=1e-3
#     )
#
#     model.train(1)
#     sys.exit()
