#!/usr/bin/env python3
"""

"""
import os
import sys
import time

import tensorflow as tf

from input_loader import PerformanceInputLoader, sequence_to_midi
from performance_rnn import PerformanceRNNModel
from transformer import TransformerModel

PROJECT_DIR = os.path.dirname(__file__)


class PerformanceModel(tf.keras.Model):
    def __init__(self,
                 inner_model,
                 input_loader,
                 model_name,
                 train_dir,
                 restore_checkpoint,
                 learning_rate):
        super().__init__(name=model_name)
        self.input_loader = input_loader
        self.train_dir = train_dir

        # TODO
        self.inner_model = inner_model

        # Adam optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # Model returns probability logits, dataset returns category indices
        self.loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.compile(optimizer=self.optimizer, loss=self.loss,
                     metrics=['accuracy'])

        checkpoint_dir = os.path.join(train_dir, 'checkpoints', model_name)
        checkpoint = tf.train.Checkpoint(model=self, optimizer=self.optimizer)
        self.batch_ctr = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.epoch_ctr = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.checkpoint_mgr = tf.train.CheckpointManager(
            checkpoint,
            directory=checkpoint_dir,
            max_to_keep=50
        )
        if restore_checkpoint:
            checkpoint.restore(self.checkpoint_mgr.latest_checkpoint)
            if self.batch_ctr.value() != 0:
                print(f'Restored checkpoint (batch {self.batch_ctr.value()}, epoch {self.epoch_ctr.value()})')
            else:
                print('Initialized model')

        self.callbacks = [TrainCallback(train_dir=train_dir)]
        self.load_time = time.localtime()

    def call(self, inputs, training=False, states=None, return_states=False):
        # TODO handle stateful / stateless inner model
        return self.inner_model(inputs, training, states, return_states)

    def train(self, epochs: int) -> None:
        """
        Note:
            Instead of simply calling self.fit() with epoch=epochs,
              we call fit() once for each training epoch.
            This is because the train dataset varies in length between epochs,
              which fit() cannot handle normally.
            The drawback is that we don't get an epoch ETA timer.
        """
        for e in range(epochs):
            self.fit(self.input_loader.dataset, epochs=1,
                     callbacks=self.callbacks,
                     validation_data=self.input_loader.test_dataset)
            print(f'Finished training epoch {e+1}/{epochs}.')

    def sample_music(self, *args, **kwargs):
        return self.inner_model.sample_music(*args, **kwargs)


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
        run_time = time.strftime('%Y.%m.%d-%H:%M:%S', self.model.load_time)
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
            for i, seq in enumerate(music):
                midi = sequence_to_midi(seq)
                midi_path = os.path.join(os.path.join(self.sample_dir, f'{self.model.name}_{step}_{i}.midi'))
                midi.save(midi_path)

        if step % self.save_checkpoint_freq == 0:
            self.model.checkpoint_mgr.save()

        self.model.batch_ctr.assign_add(1)

    def on_epoch_end(self, epoch, logs=None):
        tot_epoch = self.model.epoch_ctr.assign_add(1).value()

        print(f'\nTotal steps: {self.model.batch_ctr.value().numpy()}')
        print(f'Total epochs: {tot_epoch}\n')

        self.model.chkpt_mgr.save()

        if not logs:
            return

        val_logs = {k: v for k, v in logs.items() if k.startswith('val_')}
        if val_logs:
            with self.writer.as_default():
                for name, value in val_logs.items():
                    tf.summary.scalar(name, value, step=tot_epoch)


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
        augmentation='aug-'
    )
    # inner_model = PerformanceRNNModel(input_loader.vocab_size)
    inner_model = TransformerModel(input_loader.vocab_size, 512)

    model = PerformanceModel(inner_model, input_loader, 'outer_model', PROJECT_DIR, False, 1e-3)
    model.train(1)
    sys.exit()

