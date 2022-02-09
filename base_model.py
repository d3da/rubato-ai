#!/usr/bin/env python3
"""

"""
import os
import sys
import time

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
                 train_dir,
                 restore_checkpoint,
                 learning_rate=None,
                 adam_beta1=None,
                 adam_beta2=None,
                 adam_eps=None,
                 warmup_steps=None,
                 embed_dimension=None,
                 label_smoothing=0.0):
        super().__init__(name=model_name)
        self.input_loader = input_loader
        self.train_dir = train_dir

        # TODO abstract base class for inner model
        #      methods needed: call(), __call__(), sample_music(),
        #   properties needed: optimizer??
        # TODO config format
        self.inner_model = inner_model

        self.batch_ctr = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.epoch_ctr = tf.Variable(0, trainable=False, dtype=tf.int64)

        self.optimizer = Optimizer.create_adam_optimizer(
            learning_rate,
            adam_beta1,
            adam_beta2,
            adam_eps,
            warmup_steps,
            step_counter=self.batch_ctr,
            embed_dimension=embed_dimension
            # TODO get embed dim from inner model maybe?
            # or rather from some kinda hparam config even
        )

        self.loss = tf.losses.CategoricalCrossentropy(from_logits=True,
                                                      label_smoothing=label_smoothing)
        self.compile(optimizer=self.optimizer, loss=self.loss,
                     metrics=['accuracy'])

        checkpoint_dir = os.path.join(train_dir, 'checkpoints', model_name)
        checkpoint = tf.train.Checkpoint(model=self, optimizer=self.optimizer)
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
    def __init__(self,
                 train_dir,
                 update_freq: int = 25,
                 save_midi_freq: int = 250,
                 save_checkpoint_freq: int = 250,
                 validate_freq: int = 1000,
                 validate_batches: int = 25):
        super().__init__()

        self._train_dir = train_dir
        self._sample_dir = os.path.join(train_dir, 'train_samples')
        if not os.path.exists(self._sample_dir):
            os.mkdir(self._sample_dir)

        self._update_freq = update_freq
        self._save_midi_freq = save_midi_freq
        self._save_checkpoint_freq = save_checkpoint_freq

        self._validate_freq = validate_freq
        self._validate_batches = validate_batches

        self._batch_start_time = 0.
        self._writer = None

    def on_train_begin(self, logs=None):
        run_time = time.strftime('%Y.%m.%d-%H:%M:%S', self.model.load_time)
        log_dir = str(os.path.join('logs', self.model.name, run_time))
        self._writer = tf.summary.create_file_writer(log_dir)

    def on_train_end(self, logs=None):
        self._writer.close()

    def on_batch_begin(self, batch, logs=None):
        self._batch_start_time = time.time()

    def on_batch_end(self, batch, logs=None):
        step = self.model.batch_ctr.value()
        if logs is None:
            logs = {}

        if step % self._validate_freq == 0:
            logs = self._run_validation(self._validate_batches, logs=logs)

        if step % self._update_freq == 0:
            with self._writer.as_default():
                _batch_time = time.time() - self._batch_start_time
                tf.summary.scalar('batch_time', _batch_time, step=step)
                for key, value in logs.items():
                    tf.summary.scalar(key, value, step=step)

        if step % self._save_midi_freq == 0:
            # Generate sample
            music = self.model.sample_music()
            for i, seq in enumerate(music):
                midi = sequence_to_midi(seq)
                midi_path = os.path.join(os.path.join(self._sample_dir, f'{self.model.name}_{step}_{i}.midi'))
                midi.save(midi_path)

        if step % self._save_checkpoint_freq == 0:
            self.model.checkpoint_mgr.save()

        self.model.batch_ctr.assign_add(1)

    def on_epoch_end(self, epoch, logs=None):
        tot_epoch = self.model.epoch_ctr.assign_add(1).value()

        print(f'\nTotal steps: {self.model.batch_ctr.value().numpy()}')
        print(f'Total epochs: {tot_epoch}\n')

        self.model.checkpoint_mgr.save()

    def _run_validation(self, num_batches, logs=None):
        if logs is None:
            logs = {}
        batch_losses = []
        for i, (x, y) in enumerate(self.model.input_loader.test_dataset):
            y_hat = self.model.__call__(x, training=False)
            loss = self.model.loss.__call__(y, y_hat)
            batch_losses.append(float(loss))
            if i >= num_batches:
                break
        val_loss = np.average(batch_losses)
        logs['val_loss'] = val_loss
        return logs


if __name__ == '__main__':
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

