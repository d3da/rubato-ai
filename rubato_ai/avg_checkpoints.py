"""
.. todo::
    This class needs proper testing and improvements. I'm not sure it works at all.
"""
import os

import tensorflow as tf

from .base_model import BaseModel
from .config import default_conf, PROJECT_DIR
from .midi_processor import MidiProcessor
from .rubato_ai import RubatoAI


def load_averaged_weights(model: BaseModel, last_n=20) -> None:
    """
    Loads the last n checkpoint files and load their averaged weights.

    Using model.checkpoint_mgr.save() after calling this method is probably not a good plan,
    so I'd avoid training while/after this method is called.
    """

    checkpoints = sorted(model.checkpoint_mgr.checkpoints, key=lambda x: (len(x), x))
    checkpoints = checkpoints[-last_n:]

    sum_weights = [tf.zeros_like(w) for w in model.trainable_variables]

    for c in checkpoints:
        try:
            # TODO did we load all the right weights???
            model.checkpoint_mgr.checkpoint.read(c).expect_partial()
            print(f'Loaded checkpoint {c}', end='\r')
        except Exception as e:
            print(e)
            print(f'Could not read checkpoint file {c}')
            continue

        for i, w in enumerate(model.trainable_variables):
            sum_weights[i] += w

    for i, w in enumerate(list(model.trainable_variables)):
        w.assign(sum_weights[i] / len(checkpoints))

    print(f'\nAveraged {len(list(model.trainable_variables))} weights from {len(checkpoints)} models')

    # TODO Make sure we don't accidentally save the averaged model
    del model.checkpoint_mgr
    # model.checkpoint_mgr = None


def average_checkpoints(trainer: RubatoAI):
    load_averaged_weights(trainer.model)
    midi_processor = MidiProcessor(**default_conf)
    music = trainer.model.sample_music(sample_length=1024, temperature=0.8, verbose=True)

    midi_dir = os.path.join(PROJECT_DIR, 'samples')
    if not os.path.exists(midi_dir):
        os.mkdir(midi_dir)

    for i, seq in enumerate(music):
        events = midi_processor.indices_to_events(seq)
        midi = midi_processor.events_to_midi(events)
        midi_path = os.path.join(midi_dir, f'{trainer.model.name}_avg_{i}.midi')
        midi.save(midi_path)

    exit()  # Exit so we don't accidently continue training and overwrite a checkpoint

if __name__ == '__main__':
    p = RubatoAI('foobar', False, **default_conf)
    average_checkpoints(p)
