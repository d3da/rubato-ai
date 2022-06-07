"""
This class is messy, and will likely break with different tensorflow versions (tested on 2.7.0)

.. todo::
    - Deterministic sequence windows (no random stride but something clever)
"""
from typing import Tuple, List, Iterable, Optional

from .midi_processor import MidiProcessor
from .registry import register_param, register_links, document_registrations, PathLike, ConfDict

import os
import csv
import queue
import random
import multiprocessing

import mido
import numpy as np
import tensorflow as tf


def get_midi_filenames(csv_path: str) -> Tuple[List[str], List[str], List[str]]:

    train_set = []
    test_set = []
    validation_set = []

    with open(csv_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=',', quotechar='\"')
        for row in reader:
            split = row['split']
            # name = row['canonical_composer'] + ' // ' + row['canonical_title']
            filename = row['midi_filename']
            # duration = row['duration']
            # print(f'{split}: {name} ({float(duration):.2f})')

            if split == 'train':
                train_set.append(filename)
            elif split == 'test':
                test_set.append(filename)
            elif split == 'validation':
                validation_set.append(filename)

    return train_set, test_set, validation_set


def _augmentation_iterator(augmentation_setting: Optional[str]) -> Iterable[Tuple[int, float]]:
    """
    Go through all the desired combinations of pitch and time augmentation
    for a given augmentation setting.
    Possible values are 'aug-', 'aug+', or any other string for no augmentation.

    Generates tuples of (pitch_augmentation, time_augmentation)
    """
    if augmentation_setting == 'aug-':
        pitch_augs = range(-4, 5)
        time_augs = [0.95, 0.975, 1.0, 1.025, 1.05]
    elif augmentation_setting == 'aug+':
        pitch_augs = range(-8, 9)
        time_augs = [0.9, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.1]
    else:
        pitch_augs = [0]
        time_augs = [1.0]

    yield from zip(pitch_augs, [1.0] * len(pitch_augs))
    yield from zip([0] * len(time_augs), time_augs)


def random_augmentation(augmentation_setting: Optional[str]) -> Tuple[int, float]:
    return random.choice(list(_augmentation_iterator(augmentation_setting)))


def seq_to_windows_iterator(seq: np.array, window_size: int, min_stride: int, max_stride: int) -> Iterable[np.array]:
    # TODO the last part of the track is usually cut off :(
    i = 0
    while i < len(seq) - window_size + 1:
        yield seq[i: i + window_size]
        i += random.randrange(min_stride, max_stride + 1)


@document_registrations
@register_param('dataset_dir', PathLike, 'Path to the dataset to train on')
@register_param('dataset_csv', PathLike, 'Path to the dataset index file, relative to dataset_dir')
@register_param('sequence_length', int, '(Maximum) input sequence length')
@register_param('augmentation', str,
                'Augmentation setting in [\'\', \'aug-\', \'aug+\']')
@register_param('min_stride', int,
                'Minimum amount of tokens in distance between the beginnings of sequence windows')
@register_param('max_stride', int,
                'Maximum amount of tokens in distance between the beginnings of sequence windows')
@register_param('queue_size', int, 'Size of the window generator buffer')
@register_param('shuffle_buffer_size', int, 'Size of the window shuffle buffer')
@register_param('batch_size', int, 'Batch size to use during training')
@register_param('num_threads', int, 'Number of sequence generator processes to spawn')
@register_links({'MidiProcessor'})
class PerformanceInputLoader:
    def __init__(self, config: ConfDict):
        dataset_dir = config['dataset_dir']
        dataset_csv = os.path.join(dataset_dir, config['dataset_csv'])
        check_dataset(dataset_dir, dataset_csv)

        train, test, validation = get_midi_filenames(dataset_csv)

        window_size = config['sequence_length'] + 1

        # TODO: write static method to calculate the vocab size
        #       without instantiating a MidiProcessor (is used for nothing else here)
        self.midi_processor = MidiProcessor(config)
        self.vocab_size = self.midi_processor.vocab_size

        self.dataset = (
            tf.data.Dataset.from_generator(
                PerformanceInputLoader.threaded_window_generator,
                args=(
                    train,
                    dataset_dir,
                    config['augmentation'],
                    window_size,
                    config['min_stride'],
                    config['max_stride'],
                    config['queue_size'],
                    config['num_threads'],
                    config['time_granularity'],
                    config['piece_start'],
                    config['piece_end'],
                    config['max_silence']
                ),
                output_signature=(
                    tf.TensorSpec(shape=window_size, dtype=tf.int32)
                )).shuffle(config['shuffle_buffer_size'])  # (s+1)
                  .batch(config['batch_size'], drop_remainder=False)  # (<=b, s+1)
                  .map(self.split_x_y)  # (<=b, Tuple(s, s))
                  .prefetch(tf.data.AUTOTUNE)
        )

        # TODO we don't need the sliding window stuff on the validation set
        self.validation_dataset = (
                tf.data.Dataset.from_generator(
                    PerformanceInputLoader.threaded_window_generator,
                    args=(
                        test,
                        dataset_dir,
                        '',
                        window_size,
                        config['min_stride'],
                        config['max_stride'],
                        config['queue_size'],
                        config['num_threads'],
                        config['time_granularity'],
                        config['piece_start'],
                        config['piece_end'],
                        config['max_silence'],
                    ),
                    output_signature=(
                        tf.TensorSpec(shape=window_size, dtype=tf.int32)
                    )).batch(config['batch_size'], drop_remainder=False)
                      .map(self.split_x_y)
        )

    class SequenceProducerThread(multiprocessing.Process):
        def __init__(self,
                     path_list,
                     base_data_path,
                     augmentation,
                     buffer,
                     time_granularity,
                     piece_start,
                     piece_end,
                     max_silence):
            super().__init__()
            self.path_list = path_list
            self.base_data_path = base_data_path
            self.augmentation = augmentation
            self.buffer = buffer

            # We don't supply the config dict normally here 
            # since we are running in a different process
            self.midi_processor = MidiProcessor({
                    'time_granularity':time_granularity,
                    'piece_start':piece_start,
                    'piece_end':piece_end,
                    'max_silence':max_silence})

        def run(self):
            for path in self.path_list:
                midi = mido.MidiFile(os.path.join(self.base_data_path, path))
                pitch_augmentation, time_augmentation = random_augmentation(self.augmentation)
                events = self.midi_processor.parse_midi(midi, pitch_augmentation, time_augmentation)
                sequence = self.midi_processor.events_to_indices(events)
                self.buffer.put(np.array(list(sequence)))

    @staticmethod
    def threaded_window_generator(path_list,
                                  base_data_path,
                                  augmentation,
                                  window_size,
                                  min_stride,
                                  max_stride,
                                  queue_size,
                                  num_threads,
                                  time_granularity,
                                  piece_start,
                                  piece_end,
                                  max_silence):
        for seq in PerformanceInputLoader.threaded_sequence_generator(path_list,
                                                                      base_data_path,
                                                                      augmentation,
                                                                      queue_size,
                                                                      num_threads,
                                                                      time_granularity,
                                                                      piece_start,
                                                                      piece_end,
                                                                      max_silence):
            for win in seq_to_windows_iterator(seq, window_size, min_stride, max_stride):
                yield win

    @staticmethod
    def threaded_sequence_generator(
            path_list,
            base_data_path,
            augmentation,
            queue_size,
            num_threads,
            time_granularity,
            piece_start,
            piece_end,
            max_silence):
        random.shuffle(path_list)

        buffer = multiprocessing.Queue(queue_size)
        paths_subset = np.array_split(path_list, num_threads)
        slaves = []

        # spawn {NUM_THREADS} workers
        for i in range(num_threads):
            paths = paths_subset[i]
            slave = PerformanceInputLoader.SequenceProducerThread(
                paths, base_data_path, augmentation, buffer, time_granularity, piece_start, piece_end, max_silence
            )
            slave.start()
            slaves.append(slave)

        done = False
        while not done:
            try:
                # Return the next item in fifo queue
                seq = buffer.get(block=True, timeout=10)
                yield seq

            except queue.Empty:
                done = True
                for slave in slaves:
                    if slave.is_alive():
                        done = False

        for slave in slaves:
            slave.join()

        # No other threads are active anymore, but there may be some
        # data left in the buffer
        while not buffer.empty():
            yield buffer.get()

    def split_x_y(self, batch: tf.Tensor):
        """
        Note that only Y is converted to one_hot vectors,
        this is because the Embedding layer applied to X takes in sparse categories
        but the loss function needs one_hot encoded vectors to apply label smoothing
        """
        x = batch[:, :-1]
        y = tf.one_hot(batch[:, 1:], self.vocab_size)
        return x, y


def check_dataset(base_path: str, csv_path: str):
    if not (os.path.exists(base_path) and os.path.exists(csv_path)):
        raise FileNotFoundError(
            'You can obtain a copy of the MAESTRO dataset '
            'at https://magenta.tensorflow.org/datasets/maestro#v300')


if __name__ == '__main__':
    base_path = 'data/maestro-v3.0.0'
    csv_path = os.path.join(base_path, 'maestro-v3.0.0.csv')

    midi_conf = {'time_granularity': 100, 'piece_start': True, 'piece_end': True, 'max_silence': 6.9}
    config = {
            **midi_conf,
            'dataset_dir': 'data/maestro-v3.0.0',
            'dataset_csv': 'maestro-v3.0.0.csv',
            'sequence_length': 512,
            'augmentation': 'aug-',
            'min_stride': 256,
            'max_stride': 512,
            'queue_size': 128,
            'shuffle_buffer_size': 128,
            'batch_size': 2,
            'num_threads': 1,
    }
    input_loader = PerformanceInputLoader(config)

    for i, x in enumerate(input_loader.dataset):
        print(i, end='       \r')

    exit()

