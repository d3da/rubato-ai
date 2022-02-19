#!/usr/bin/env python3
from typing import Tuple, List, Iterable, Optional

from midi_reader import events_to_midi, midi_to_events, Event

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


def sequence_to_midi(seq: List[int]) -> mido.MidiFile:
    return events_to_midi(map(Event.from_category, seq))


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


def file_to_seq(path: str, base_data_path: str,
                augmentation: Optional[str] = None) -> np.array:
    midi = mido.MidiFile(os.path.join(base_data_path, path))
    pitch_augmentation, time_augmentation = random_augmentation(augmentation)
    events = midi_to_events(midi, pitch_augmentation, time_augmentation)
    return np.array(list(map(lambda e: e.category, events)))


def seq_to_windows_iterator(seq: np.array, window_size: int, min_stride: int, max_stride: int) -> Iterable[np.array]:
    # TODO the last part of the track is usually cut off :(
    i = 0
    while i < len(seq) - window_size + 1:
        yield seq[i: i + window_size]
        i += random.randrange(min_stride, max_stride + 1)


class PerformanceInputLoader:

    # def __init__(self, base_data_path: str, csv_path: str,
    #              sequence_length: int, min_stride: int, max_stride: int,
    #              batch_size: int, augmentation: Optional[str],
    #              shuffle_buffer_size, queue_size, num_threads):
    def __init__(self, dataset_base_path, dataset_csv, **config):

        check_dataset(dataset_base_path, dataset_csv)

        train, test, validation = get_midi_filenames(dataset_csv)

        window_size = config['sequence_length'] + 1

        self.vocab_size = Event.vocab_size

        self.dataset = (
            tf.data.Dataset.from_generator(
                PerformanceInputLoader.threaded_window_generator,
                args=(
                    train,
                    dataset_base_path,
                    config['augmentation'],
                    window_size,
                    config['min_stride'],
                    config['max_stride'],
                    config['queue_size'],
                    config['num_threads']
                ),
                output_signature=(
                    tf.TensorSpec(shape=window_size, dtype=tf.int32)
                )).shuffle(config['shuffle_buffer_size'])  # (s+1)
                  .batch(config['batch_size'], drop_remainder=False)  # (<=b, s+1)
                  .map(PerformanceInputLoader.split_x_y)  # (<=b, Tuple(s, s))
                  .prefetch(tf.data.AUTOTUNE)
        )

        # TODO we don't need the sliding window stuff on the validation set
        self.validation_dataset = (
                tf.data.Dataset.from_generator(
                    PerformanceInputLoader.threaded_window_generator,
                    args=(
                        test,
                        dataset_base_path,
                        '',
                        window_size,
                        config['min_stride'],
                        config['max_stride'],
                        config['queue_size'],
                        config['num_threads']
                    ),
                    output_signature=(
                        tf.TensorSpec(shape=window_size, dtype=tf.int32)
                    )).batch(config['batch_size'], drop_remainder=False)
                      .map(PerformanceInputLoader.split_x_y)
        )

    class SequenceProducerThread(multiprocessing.Process):
        def __init__(self, path_list, base_data_path, augmentation, buffer):
            super().__init__()
            self.path_list = path_list
            self.base_data_path = base_data_path
            self.augmentation = augmentation
            self.buffer = buffer

        def run(self):
            for path in self.path_list:
                seq = file_to_seq(path, self.base_data_path, self.augmentation)
                self.buffer.put(seq)

    @staticmethod
    def threaded_window_generator(path_list,
                                  base_data_path,
                                  augmentation,
                                  window_size,
                                  min_stride,
                                  max_stride,
                                  queue_size,
                                  num_threads):
        for seq in PerformanceInputLoader.threaded_sequence_generator(path_list,
                                                                      base_data_path,
                                                                      augmentation,
                                                                      queue_size,
                                                                      num_threads):
            for win in seq_to_windows_iterator(seq, window_size, min_stride, max_stride):
                yield win

    @staticmethod
    def threaded_sequence_generator(
            path_list,
            base_data_path,
            augmentation,
            queue_size,
            num_threads):
        random.shuffle(path_list)

        buffer = multiprocessing.Queue(queue_size)
        paths_subset = np.array_split(path_list, num_threads)
        slaves = []

        # spawn {NUM_THREADS} workers
        for i in range(num_threads):
            paths = paths_subset[i]
            slave = PerformanceInputLoader.SequenceProducerThread(paths, base_data_path, augmentation, buffer)
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

    @staticmethod
    def split_x_y(batch: tf.Tensor):
        """
        Note that only Y is converted to one_hot vectors,
        this is because the Embedding layer applied to X takes in sparse categories
        but the loss function needs one_hot encoded vectors to apply label smoothing
        """
        x = batch[:, :-1]
        y = tf.one_hot(batch[:, 1:], Event.vocab_size)
        return x, y


def check_dataset(base_path: str, csv_path: str):
    if not (os.path.exists(base_path) and os.path.exists(csv_path)):
        raise FileNotFoundError(
            'You can obtain a copy of the MAESTRO dataset'
            'at https://magenta.tensorflow.org/datasets/maestro#v300')


if __name__ == '__main__':
    base_path = 'data/maestro-v3.0.0'
    csv_path = os.path.join(base_path, 'maestro-v3.0.0.csv')

    exit()

