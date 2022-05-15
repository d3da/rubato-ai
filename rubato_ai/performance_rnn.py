"""
This Time with Feeling: Learning Expressive Musical Performance
by Sageev Oore, Ian Simon, Sander Dieleman, Douglas Eck, Karen Simonyan
https://arxiv.org/abs/1808.03715v1
"""
import time

import tensorflow as tf

from .base_model import PerformanceModel
from .registry import register_param, register_links


@register_param('rnn_units', int, 'Hidden dimension of the RNN')
@register_param('drop_rate', float,
                'Dropout rate to apply to RNN layers')
@register_links({'PerformanceModel'})
class PerformanceRNNModel(PerformanceModel):
    """
    seq_length: 512a
    lstm_h_dim: 512b
    vocab_size: 413
    batch_size: 64
    learn_rate: 1e-3

    Input   -> (64, 512a)
    One_Hot -> (64, 512a, 413)
    LSTM    -> (64, 512a, 512b)
    LSTM    -> (64, 512a, 512b)
    LSTM    -> (64, 512a, 512b)
    Dense   -> (64, 512a, 413)  (no activation)

    TODO parametric number of layers
    TODO select 'lstm' | 'gru'
    TODO test

    TODO it doesn't work with ``config['mixed_precision']``
    """
    def __init__(self,
                 model_name,
                 input_loader,
                 restore_checkpoint,
                 **config):
        super().__init__(model_name, input_loader, restore_checkpoint, **config)
        self._vocab_size = self.input_loader.vocab_size
        self._rnn_units = config['rnn_units']
        self._drop_rate = config['drop_rate']

        # Model layers
        self.lstm1 = tf.keras.layers.LSTM(self._rnn_units,
                                          return_sequences=True,
                                          return_state=True,
                                          dropout=self._drop_rate)
        self.lstm2 = tf.keras.layers.LSTM(self._rnn_units,
                                          return_sequences=True,
                                          return_state=True,
                                          dropout=self._drop_rate)
        self.lstm3 = tf.keras.layers.LSTM(self._rnn_units,
                                          return_sequences=True,
                                          return_state=True,
                                          dropout=self._drop_rate)
        self.dense = tf.keras.layers.Dense(self._vocab_size)

    def call(self, inputs, training=False, states=None, return_states=False):
        """
        Use builtin self.__call__() instead
        """
        x = tf.one_hot(inputs, self._vocab_size)
        if states is None:
            s_1, c_1 = self.lstm1.get_initial_state(x)
            s_2, c_2 = self.lstm2.get_initial_state(x)
            s_3, c_3 = self.lstm3.get_initial_state(x)
        else:
            s_1, c_1, s_2, c_2, s_3, c_3 = states
        x, s_1, c_1 = self.lstm1(x, training=training, initial_state=[s_1, c_1])
        x, s_2, c_2 = self.lstm2(x, training=training, initial_state=[s_2, c_2])
        x, s_3, c_3 = self.lstm3(x, training=training, initial_state=[s_3, c_3])
        x = self.dense(x, training=training)
        if return_states:
            return x, (s_1, c_1, s_2, c_2, s_3, c_3)
        return x

    @tf.function
    def generate_step(self, inputs, states, temperature):
        predicted_logits, states = self(inputs=inputs, states=states, return_states=True)
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits *= temperature
        predicted_categories = tf.random.categorical(predicted_logits, num_samples=1, dtype=tf.int32)
        return predicted_categories, states

    # todo primers etc
    @tf.function
    def sample_music(self, start_event_category=0, sample_length=512, num_seqs=2, temperature=1.0, verbose=False):
        states = None
        next_category = tf.constant([start_event_category]*num_seqs, shape=(num_seqs, 1), dtype=tf.int32)
        temperature = tf.constant(temperature, dtype=tf.float32)
        result = next_category[:]
        start = time.time()
        for i in range(sample_length):
            if verbose:
                print(f'Sampling... {i}/{sample_length}', end='\r')
            next_category, states = self.generate_step(
                    next_category, states, temperature)
            result = tf.concat([result, next_category], axis=1)
        if verbose:
            print(f'Sampled {sample_length} tokens in {time.time()-start:.2f} s.')
        return result
