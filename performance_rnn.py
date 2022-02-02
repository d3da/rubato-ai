#!/usr/bin/env python3
"""
This Time with Feeling: Learning Expressive Musical Performance
by Sageev Oore, Ian Simon, Sander Dieleman, Douglas Eck, Karen Simonyan
https://arxiv.org/abs/1808.03715v1

Implemented by Daan Dieperink
<d.dieperink@student.utwente.nl>


TODO:
    Model:
        Add validation data to model.fit
        sampling temp / beamsearch etc
    Misc:
        Remove relative paths, make all relative to __dir__

    Train!

Future improvements:
    Argparse for cli use
    Optimization/profiling
    Playback to alsa sequencer
"""
from input_loader import PerformanceInputLoader
from input_loader import sequence_to_midi

import os
import sys
import time

import tensorflow as tf


PROJECT_DIR = os.path.dirname(__file__)


class PerformanceRNNModel(tf.keras.Model):
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
    """
    def __init__(self,
                 input_loader: PerformanceInputLoader,
                 model_name: str = 'performance_rnn',
                 train_dir: str = PROJECT_DIR,
                 restore_chkpt: bool = True,
                 rnn_units: int = 512,
                 learning_rate: float = 1e-3,
                 dropout: float = 0.0):
        super().__init__(self, name=model_name)
        self.input_loader = input_loader
        self.vocab_size = input_loader.vocab_size
        self.train_dir = train_dir

        # Model layers
        self.lstm1 = tf.keras.layers.LSTM(rnn_units,
                                          return_sequences=True,
                                          return_state=True,
                                          dropout=dropout)
        self.lstm2 = tf.keras.layers.LSTM(rnn_units,
                                          return_sequences=True,
                                          return_state=True,
                                          dropout=dropout)
        self.lstm3 = tf.keras.layers.LSTM(rnn_units,
                                          return_sequences=True,
                                          return_state=True,
                                          dropout=dropout)
        self.dense = tf.keras.layers.Dense(self.vocab_size)

        # Optimize using stochastic gradient descent
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        # Model returns probability logits, dataset returns category indices
        self.loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.compile(optimizer=self.optimizer, loss=self.loss,
                     metrics=['accuracy'])

        # Setup checkpoints
        chkpt_dir = os.path.join(train_dir, 'checkpoints', model_name)
        chkpt = tf.train.Checkpoint(model=self, optimizer=self.optimizer)
        self.batch_ctr = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.chkpt_mgr = tf.train.CheckpointManager(
                chkpt,
                directory=chkpt_dir,
                max_to_keep=100,
                step_counter=self.batch_ctr)
        if restore_chkpt:
            chkpt.restore(self.chkpt_mgr.latest_checkpoint)
            if self.batch_ctr.value() != 0:
                print(f'Restored checkpoint (batch {self.batch_ctr.value()})')
            else:
                print('Initialized model')

        self.callbacks = [TrainCallback(train_dir=train_dir)]

    def call(self, inputs, training=False, states=None, return_states=False):
        """
        Use builtin self.__call__() instead
        """
        x = tf.one_hot(inputs, self.vocab_size)
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

    def train(self, epochs):
        # TODO validation metrics
        return self.fit(self.input_loader.dataset, epochs=epochs,
                        callbacks=self.callbacks,
                        validation_data=self.input_loader.test_dataset)

    @tf.function
    def generate_step(self, inputs, states, temperature):
        predicted_logits, states = self(inputs=inputs, states=states, return_states=True)
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits *= temperature
        predicted_categories = tf.random.categorical(predicted_logits, num_samples=1)
        return predicted_categories, states

    # todo provide interface for sampling temperature / beam search etc
    # todo generate multiple samples at once
    def sample_music(self, start_event_category=0, sample_length=512, temperature=1.0):
        states = None
        next_category = tf.constant([start_event_category], shape=(1, 1))
        temperature = tf.constant(temperature, dtype=tf.float32)
        result = [next_category]
        for _ in range(sample_length):
            next_category, states = self.generate_step(
                    next_category, states, temperature)
            result.append(next_category)
        result = [r.numpy()[0][0] for r in result]
        return result


class TrainCallback(tf.keras.callbacks.Callback):
    """
    Custom callback that provides "improvements" over the default
        tf.keras.callbacks.TensorBoard in addition to handling
        checkpoint saving and sample generation.

    This callback keeps track of a global step (batch) counter
        that persists between checkpoint saves/loads, allowing
        tensorboard graphs to span multiple runs.

    """
    def __init__(self,
                 train_dir,
                 update_freq: int = 25,
                 save_midi_freq: int = 50,
                 save_checkpoint_freq: int = 100,
                 write_steps_per_second: bool = True):
        super().__init__()
        # TODO global epoch as well?

        self.train_dir = train_dir
        self.sample_dir = os.path.join(train_dir, 'train_samples')
        if not os.path.exists(self.sample_dir):
            os.mkdir(self.sample_dir)

        self.update_freq = update_freq
        self.save_midi_freq = save_midi_freq
        self.save_checkpoint_freq = save_checkpoint_freq
        self.write_steps_per_second = write_steps_per_second

        self._batch_start_time = 0.
        self.writer = None

    def on_train_begin(self, logs=None):
        run_time = time.strftime('%Y.%m.%d-%H:%M:%S', time.localtime())
        log_dir = str(os.path.join('logs', self.model.name, run_time))
        self.writer = tf.summary.create_file_writer(log_dir)

    def on_train_end(self, logs=None):
        self.writer.close()

    def on_batch_begin(self, batch, logs=None):
        if self.write_steps_per_second:
            self._batch_start_time = time.time()

    def on_batch_end(self, batch, logs=None):
        step = self.model.batch_ctr.value()
        if step % self.update_freq == 0:
            with self.writer.as_default():
                if self.write_steps_per_second:
                    _batch_time = time.time() - self._batch_start_time
                    tf.summary.scalar('batch_time', _batch_time, step=step)
                for key, value in logs.items():
                    tf.summary.scalar(key, value, step=step)

        if step % self.save_midi_freq == 0:
            # Generate sample
            music = self.model.sample_music()
            midi = sequence_to_midi(music)
            midi_path = os.path.join(os.path.join(self.sample_dir, f'sample_batch_{step}.midi'))
            midi.save(midi_path)

        if step % self.save_checkpoint_freq == 0:
            self.model.chkpt_mgr.save()

        self.model.batch_ctr.assign_add(1)

    def on_epoch_end(self, epoch, logs=None):
        print(f'\nTotal steps: {self.model.batch_ctr.value().numpy()}\n')


if __name__ == '__main__':
    dataset_base = os.path.join(PROJECT_DIR, 'data/maestro-v3.0.0')
    dataset_csv = os.path.join(dataset_base, 'maestro-v3.0.0.csv')

    input_loader = PerformanceInputLoader(
        dataset_base,
        dataset_csv,
        sequence_length=512,
        min_stride=128,
        max_stride=256,
        batch_size=64,
        augmentation='aug-')

    model = PerformanceRNNModel(
        input_loader=input_loader,
        model_name='performance_rnn',
        train_dir=PROJECT_DIR,
        restore_chkpt=True,
        rnn_units=512,
        learning_rate=1e-3,
        dropout=0.0
    )

    model.train(500)
    sys.exit()
