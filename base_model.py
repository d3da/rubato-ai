#!/usr/bin/env python3
"""

"""
import os
import time
from typing import Generator

import numpy as np
import tensorflow as tf

from midi_processor import MidiProcessor
from optimizer import Optimizer

from registry import register_param, register_links, PathLike

PROJECT_DIR = os.path.dirname(__file__)


@register_param('train_dir', PathLike, PROJECT_DIR,
                'Path for saving checkpoints, tensorboard logs and samples')
@register_param('kept_checkpoints', int, 50,
                'Number of checkpoints to save in checkpoint directory')
@register_param('label_smoothing', float, 0.1,
                'Amount of label smoothing regularization to apply to training examples')
@register_links({'Optimizer', 'TrainCallback'})
class PerformanceModel(tf.keras.Model):
    """
    Base class inherited by TransformerModel and PerformanceRNNModel.
    This class can be considered abstract and is not instantiated directly.

    This class handles the following:
        - Setup the optimizer
        - Run the train() loop
        - Keep persistent batch / epoch counters
        - Save checkpoints
    """
    def __init__(self,
                 model_name,
                 input_loader,
                 restore_checkpoint,
                 **config):
        super().__init__(name=model_name)
        self.input_loader = input_loader
        self.train_dir = config['train_dir']

        self._batch_ctr = tf.Variable(0, trainable=False, dtype=tf.int64)
        self._epoch_ctr = tf.Variable(0, trainable=False, dtype=tf.int64)

        self.optimizer = Optimizer.create(step_counter=self._batch_ctr, **config)

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
            if self.batch_count != 0:
                print(f'Restored checkpoint (batch {self.batch_count}, epoch {self.epoch_count})')
            else:
                print('Initialized model (we\'re at batch zero)')

        self.callbacks = [TrainCallback(**config)]
        self.load_time = time.localtime()

    @property
    def batch_count(self):
        return self._batch_ctr.value().numpy()

    @property
    def epoch_count(self):
        return self._epoch_ctr.value().numpy()

    def increment_batch(self):
        return self._batch_ctr.assign_add(1).value().numpy()

    def increment_epoch(self):
        return self._epoch_ctr.assign_add(1).value().numpy()

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


@register_param('train_dir', 'str or os.PathLike', PROJECT_DIR,
                'Path for saving checkpoints, tensorboard logs and samples')
@register_param('tensorboard_update_freq', 'int', 50,
                'Number of batches between tensorboard updates')
@register_param('sample_midi_freq', 'int', 250,
                'Batches between saving midi sample to disk')
@register_param('sample_midi_length', 'int', 512,
                'Number of tokens to sample')
@register_param('validation_freq', 'int', 1000,
                'Batches between evaluating validation data')
@register_param('validation_batches', 'int', 25,
                'Batches to evaluate validation data for')
@register_param('save_checkpoint_freq', 'int', 500,
                'Batches between saving checkpoint to disk')
@register_param('kept_checkpoints', 'int', 50,
                'Number of checkpoints to save in checkpoint directory')
@register_links({'MidiProcessor'})
class TrainCallback(tf.keras.callbacks.Callback):
    """
    Custom callback that provides "improvements" over the default
        tf.keras.callbacks.TensorBoard in addition to handling
        checkpoint saving and sample generation.

    This callback keeps track of a global step (batch) counter
        that persists between checkpoint saves/loads, allowing
        tensorboard graphs to span multiple runs.

    TODO this class has become bloated and should be split up into multiple Callbacks
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

        self._midi_processor = MidiProcessor(**config)

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
        assert self._writer is not None
        self._writer.close()

    def on_batch_begin(self, batch, logs=None):
        self._batch_start_time = time.time()

    def on_batch_end(self, batch, logs=None):
        _batch_time = time.time() - self._batch_start_time
        step = self.model.batch_count

        if batch == 0:
            self.model.increment_batch()
            return

        if logs is None:
            logs = {}

        if step % self._validation_freq == 0:
            logs = self._run_validation(self._validation_batches, logs)

        if step % self._tensorboard_update_freq == 0:
            assert self._writer is not None
            with self._writer.as_default():
                tf.summary.scalar('batch_time', _batch_time, step=step)
                for key, value in logs.items():
                    tf.summary.scalar(key, value, step=step)
            self.model.reset_metrics()

        if step % self._save_midi_freq == 0:
            # Generate sample
            music = self.model.sample_music()
            for i, seq in enumerate(music):
                events = self._midi_processor.indices_to_events(seq)
                midi = self._midi_processor.events_to_midi(events)
                assert self._sample_subdir is not None
                midi_path = os.path.join(os.path.join(self._sample_subdir, f'{self.model.name}_{step}_{i}.midi'))
                midi.save(midi_path)

        if step % self._save_checkpoint_freq == 0:
            self.model.checkpoint_mgr.save()

        self.model.increment_batch()

    def on_epoch_end(self, epoch, logs=None):
        tot_epoch = self.model.increment_epoch()

        print(f'\nTotal steps: {self.model.batch_count}')
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
    exit()

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
    # inner_model = TransformerModel(
    #     vocab_size=input_loader.vocab_size,
    #     sequence_length=2048,
    #     num_layers=8,
    #     drop_rate=0.2,
    #     embed_dim=384,
    #     attn_heads=8,
    #     ff_dim=1024,
    #     attn_dim=512
    # )
    #
    # model = PerformanceModel(
    #     inner_model,
    #     input_loader,
    #     'outer_model',  #todo don't allow 2 different model types wiith same name
    #     PROJECT_DIR,
    #     restore_checkpoint=True,
    #
    #     # # Simon & Oore (2018)
    #     # learning_rate=1e-3,
    #
    #     # Vaswani et al. (2017)
    #     learning_rate=None,
    #     adam_beta1=0.9,
    #     adam_beta2=0.98,
    #     adam_eps=1e-9,
    #     warmup_steps=4000,
    #     # embed_dimension=512,
    #     label_smoothing=0.1,
    #
    #     # Huang et al. (2018)
    #     embed_dimension=384,
    # )
    #
    # model.__call__(tf.zeros((1, 2048), dtype=tf.int32))
    # model.summary()
    # model.train(1)
    # sys.exit()
