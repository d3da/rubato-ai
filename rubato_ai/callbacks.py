import os
import time

import numpy as np
import tensorflow as tf

from .registry import register_param, register_links, PathLike
from .midi_processor import MidiProcessor

from typing import Generator


@register_param('train_dir', PathLike,
                'Path for saving checkpoints, tensorboard logs and samples')
@register_param('tensorboard_update_freq', int,
                'Number of batches between tensorboard updates')
@register_param('sample_midi_freq', int,
                'Batches between saving midi sample to disk')
@register_param('sample_midi_length', int,
                'Number of tokens to sample')
@register_param('validation_freq', int,
                'Batches between evaluating validation data')
@register_param('validation_batches', int,
                'Batches to evaluate validation data for')
@register_param('save_checkpoint_freq', int,
                'Batches between saving checkpoint to disk')
@register_param('kept_checkpoints', int,
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
    and moved to a different file.

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
