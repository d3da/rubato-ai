#!/usr/bin/env python3
"""

"""
import os
import sys
import time
from typing import Generator

import numpy as np
import tensorflow as tf

from input_loader import PerformanceInputLoader, sequence_to_midi
from optimizer import Optimizer
from performance_rnn import PerformanceRNNModel
from transformer import TransformerModel

PROJECT_DIR = os.path.dirname(__file__)


class PerformanceModel(tf.keras.Model):
    def __init__(self,
                 inner_model,
                 input_loader,
                 model_name,
                 restore_checkpoint,
                 **config):
        super().__init__(name=model_name)
        self.input_loader = input_loader
        self.train_dir = config['train_dir']

        # TODO abstract base class for inner model
        #      methods needed: call(), __call__(), sample_music(),
        #   properties needed: optimizer??
        self.inner_model = inner_model

        self.batch_ctr = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.epoch_ctr = tf.Variable(0, trainable=False, dtype=tf.int64)

        self.optimizer = Optimizer.create_adam_optimizer(step_counter=self.batch_ctr, **config)

        self.loss = tf.losses.CategoricalCrossentropy(from_logits=True,
                                                      label_smoothing=config['label_smoothing'])
        self.compile(optimizer=self.optimizer, loss=self.loss,
                     metrics=['accuracy'])

        checkpoint_dir = os.path.join(self.train_dir, 'checkpoints', model_name)
        checkpoint = tf.train.Checkpoint(model=self, optimizer=self.optimizer)
        self.checkpoint_mgr = tf.train.CheckpointManager(
            checkpoint,
            directory=checkpoint_dir,
            max_to_keep=config['kept_checkpoints']
        )
        if restore_checkpoint:
            checkpoint.restore(self.checkpoint_mgr.latest_checkpoint)
            if self.batch_ctr.value() != 0:
                print(f'Restored checkpoint (batch {self.batch_ctr.value()}, epoch {self.epoch_ctr.value()})')
            else:
                print('Initialized model (we\'re at batch zero)')

        self.callbacks = [TrainCallback(**config)]
        self.load_time = time.localtime()

    def call(self, inputs, training=False):
        return self.inner_model.__call__(inputs, training=training)

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
                     callbacks=self.callbacks)
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
    def __init__(self, **config):
        super().__init__()

        self._train_dir = config['train_dir']

        self._tensorboard_update_freq = config['tensorboard_update_freq']
        self._save_midi_freq = config['sample_midi_freq']
        self._save_checkpoint_freq = config['save_checkpoint_freq']

        self._validation_freq = config['validation_freq']
        self._validation_batches = config['validation_batches']
        self._validation_set = self._validation_generator()

        self._batch_start_time = 0.
        self._writer = None  # Defer instantiating writer and sample subdirectories before training
        self._sample_subdir = None  # to avoid making empty subdirectories when not training

    def on_train_begin(self, logs=None):
        run_time = time.strftime('%Y.%m.%d-%H:%M:%S', self.model.load_time)
        log_dir = str(os.path.join('logs', self.model.name, run_time))
        self._writer = tf.summary.create_file_writer(log_dir)

        sample_dir = os.path.join(self._train_dir, 'train_samples')
        self._sample_subdir = os.path.join(sample_dir, self.model.name)
        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)
        if not os.path.exists(self._sample_subdir):
            os.mkdir(self._sample_subdir)

    def on_train_end(self, logs=None):
        self._writer.close()

    def on_batch_begin(self, batch, logs=None):
        self._batch_start_time = time.time()

    def on_batch_end(self, batch, logs=None):
        _batch_time = time.time() - self._batch_start_time
        step = self.model.batch_ctr.value()

        if step == 0:
            self.model.batch_ctr.assign_add(1)
            return

        if logs is None:
            logs = {}

        if step % self._validation_freq == 0:
            logs = self._run_validation(self._validation_batches, logs)

        if step % self._tensorboard_update_freq == 0:
            with self._writer.as_default():
                tf.summary.scalar('batch_time', _batch_time, step=step)
                for key, value in logs.items():
                    tf.summary.scalar(key, value, step=step)
            self.model.reset_metrics()

        if step % self._save_midi_freq == 0:
            # Generate sample
            music = self.model.sample_music()
            for i, seq in enumerate(music):
                midi = sequence_to_midi(seq)
                midi_path = os.path.join(os.path.join(self._sample_subdir, f'{self.model.name}_{step}_{i}.midi'))
                midi.save(midi_path)

        if step % self._save_checkpoint_freq == 0:
            self.model.checkpoint_mgr.save()

        self.model.batch_ctr.assign_add(1)

    def on_epoch_end(self, epoch, logs=None):
        tot_epoch = self.model.epoch_ctr.assign_add(1).value()

        print(f'\nTotal steps: {self.model.batch_ctr.value().numpy()}')
        print(f'Total epochs: {tot_epoch}\n')

        self.model.checkpoint_mgr.save()

    def _run_validation(self, num_batches, logs):
        batch_losses = []
        for _ in range(num_batches):
            x, y = self._validation_set.__next__()
            y_hat = self.model.__call__(x, training=False)
            loss = self.model.loss.__call__(y, y_hat)
            batch_losses.append(float(loss))
        val_loss = np.average(batch_losses)
        logs['val_loss'] = val_loss
        return logs

    def _validation_generator(self) -> Generator:
        """
        (Temporary?) Workaround:
        Everytime that input_loader.validation_dataset.__iter__() is called,
            new processes are spawned to generate the data (by processing midi files).
        These processes accumulate and cause a memory leak slash slowly detonating fork bomb,
            when __iter__ is repeatedly called without iterating through the entire epoch.
        Hence, we wrap the dataset in another generator (this function) which reuses an iterator
            until it is entirely consumed, allowing the processes to be cleaned up properly.
        """
        while True:
            for x, y in self.model.input_loader.validation_dataset:
                yield x, y


if __name__ == '__main__':
    exit(print('Run config.py instead'))
    dataset_base = os.path.join(PROJECT_DIR, 'data/maestro-v3.0.0')
    dataset_csv = os.path.join(dataset_base, 'maestro-v3.0.0.csv')

    input_loader = PerformanceInputLoader(
        dataset_base,
        dataset_csv,
        sequence_length=2048,
        min_stride=1024,
        max_stride=2048,
        batch_size=1,
        augmentation='aug-'
    )

    # # Simon & Oore (2018)
    # inner_model = PerformanceRNNModel(
    #     vocab_size=input_loader.vocab_size,
    #     rnn_units=512,
    #     dropout=0.0
    # )

    # # Vaswani 2017
    # inner_model = TransformerModel(
    #     vocab_size=input_loader.vocab_size,
    #     sequence_length=512,
    #     num_layers=6,
    #     drop_rate=0.1,
    #     embed_dim=512,
    #     attn_heads = 8,
    #     ff_dim = 2048,
    #     attn_dim=None
    # )

    # Huang 2018 (Baseline transformer)
    inner_model = TransformerModel(
        vocab_size=input_loader.vocab_size,
        sequence_length=2048,
        num_layers=8,
        drop_rate=0.2,
        embed_dim=384,
        attn_heads=8,
        ff_dim=1024,
        attn_dim=512
    )

    model = PerformanceModel(
        inner_model,
        input_loader,
        'outer_model',  #todo don't allow 2 different model types wiith same name
        PROJECT_DIR,
        restore_checkpoint=True,

        # # Simon & Oore (2018)
        # learning_rate=1e-3,

        # Vaswani et al. (2017)
        learning_rate=None,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_eps=1e-9,
        warmup_steps=4000,
        # embed_dimension=512,
        label_smoothing=0.1,

        # Huang et al. (2018)
        embed_dimension=384,
    )

    model.__call__(tf.zeros((1, 2048), dtype=tf.int32))
    model.summary()
    model.train(1)
    sys.exit()

