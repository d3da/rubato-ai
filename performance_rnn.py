#!/usr/bin/env python3
"""
This Time with Feeling: Learning Expressive Musical Performance
by Sageev Oore, Ian Simon, Sander Dieleman, Douglas Eck, Karen Simonyan
https://arxiv.org/abs/1808.03715v1

Implemented by Daan Dieperink
<d.dieperink@student.utwente.nl>


TODO:
    Model:
        midi file generation and playback
        checkpoints, tensorboard
        Custom training loop and callbacks
        Add validation data to model.fit
    Input loader:
        Dataset shuffling (or randomization at least)
        Data augmentation
        Save preprocessed dataset to disk

    Train!

Future improvements:
    Argparse for cli use
    Optimization/profiling
    Playback to alsa sequencer
"""
import os.path

from input_loader import PerformanceInputLoader
from input_loader import sequence_to_midi

import sys
import time
import pdb

import tensorflow as tf
import numpy as np



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
    def __init__(self, input_loader: PerformanceInputLoader,
                 model_name: str, checkpoint_dir: str, restore_chkpt: bool,
                 rnn_units: int = 512, learning_rate: float = 1e-3,
                 dropout: float = 0.0):
        super().__init__(self, name=model_name)
        self.input_loader = input_loader
        self.vocab_size = input_loader.vocab_size

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

        # Adam optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # Model returns probability logits, dataset returns category indices
        self.loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.compile(optimizer=self.optimizer, loss=self.loss)

        # Setup callbacks for during training
        self.callbacks = [
                TrainCallback(),
                tf.keras.callbacks.TensorBoard(
                    profile_batch=(5, 20),
                    update_freq=5)]

        # Setup checkpoints 
        self.chkpt = tf.train.Checkpoint(model=self, optimizer=self.optimizer)
        self.batch_ctr = tf.Variable(1, trainable=False)
        self.chkpt_mgr = tf.train.CheckpointManager(
                self.chkpt,
                directory=checkpoint_dir,
                max_to_keep=50,
                step_counter=self.batch_ctr,
                checkpoint_interval=5)
        if restore_chkpt:
            self.chkpt.restore(self.chkpt_mgr.latest_checkpoint)
            print(f'Restored checkpoint (batch {self.batch_ctr.value()})')


    def call(self, inputs, training=False, states=None, return_states=False):
        """
        TODO handle cell state as well as activation state for sampling
        """
        x = tf.one_hot(inputs, self.vocab_size)
        if states is None:
            state_1, cell_1 = self.lstm1.get_initial_state(x)
            state_2, cell_2 = self.lstm2.get_initial_state(x)
            state_3, cell_3 = self.lstm3.get_initial_state(x)
        else:
            state_1, cell_1, state_2, cell_2, state_3, cell_3 = states
        x, state_1, cell_1 = self.lstm1(x, training=training, initial_state=[state_1, cell_1])
        x, state_2, cell_2 = self.lstm2(x, training=training, initial_state=[state_2, cell_2])
        x, state_3, cell_3 = self.lstm3(x, training=training, initial_state=[state_3, cell_3])
        x = self.dense(x, training=training)
        if return_states:
            return x, (state_1, cell_1, state_2, cell_2, state_3, cell_3)
        return x

    def train(self, epochs):
        return self.fit(self.input_loader.dataset, epochs=epochs, callbacks=self.callbacks)


    @tf.function
    def generate_step(self, inputs, states):
        predicted_logits, states = self(inputs=inputs, states=states, return_states=True)
        predicted_logits = predicted_logits[:, -1, :]
        predicted_categories = tf.random.categorical(predicted_logits, num_samples=1)
        return predicted_categories, states

    def sample_music(self, start_event_category=0, sample_length=512):
        states = None
        next_category = tf.constant([start_event_category], shape=(1, 1))
        result = [next_category]
        for _ in range(sample_length):
            next_category, states = self.generate_step(
                    next_category, states)
            result.append(next_category)
        result = [r.numpy()[0][0] for r in result]
        return result


class TrainCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        self.model.batch_ctr.assign_add(1)
        b = self.model.batch_ctr.value()
        if batch % 50 == 0:
            # Generate sample
            music = self.model.sample_music()
            midi = sequence_to_midi(music)
            midi.save(f'./results/out-b{b}.midi')
            print(f'\nSaved midi (batch {b})')
            # Save ckpt
            self.model.chkpt_mgr.save()


if __name__ == '__main__':
    project_dir = os.path.dirname(__file__)
    dataset_base = os.path.join(project_dir, 'data/maestro-v3.0.0')
    dataset_csv = os.path.join(dataset_base, 'maestro-v3.0.0.csv')

    input_loader = PerformanceInputLoader(
        dataset_base,
        dataset_csv,
        sequence_length=512,
        min_stride=128,
        max_stride=256,
        batch_size=64,
        augmentation='aug+')
    model = PerformanceRNNModel(
        input_loader,
        'performance_rnn-0.1',
        './ckpt',
        restore_chkpt=True)

    model.train(100)
    sys.exit()
