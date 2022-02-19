import os

import tensorflow as tf

from base_model import PerformanceModel
from config import load_model_from_config, default_conf, PROJECT_DIR
from input_loader import sequence_to_midi


def load_averaged_weights(model: PerformanceModel, last_n=20) -> None:
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


if __name__ == '__main__':
    model = load_model_from_config(default_conf)
    load_averaged_weights(model)
    music = model.sample_music(sample_length=1024, temperature=0.8, verbose=True)

    midi_dir = os.path.join(PROJECT_DIR, 'samples')
    if not os.path.exists(midi_dir):
        os.mkdir(midi_dir)

    for i, seq in enumerate(music):
        midi = sequence_to_midi(seq)
        midi_path = os.path.join(midi_dir, f'{model.name}_avg_{i}.midi')
        midi.save(midi_path)
    exit()

else:
    raise ImportError('Importing avg_checkpoints.py not yet supported... '
                      '(for fear of overwriting checkpoints)')
