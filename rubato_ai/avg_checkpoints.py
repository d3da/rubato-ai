"""
.. todo::
    This class needs proper testing and improvements. I'm not sure it works at all.

.. todo::
    Add interface to this class to __main__
"""
import tensorflow as tf

from .base_model import BaseModel

def load_averaged_weights(model: BaseModel, last_n: int = 20) -> None:
    """
    Loads the last n checkpoint files and load their averaged weights.

    Using model.checkpoint_mgr.save() after calling this method is probably not a good plan,
    so I'd avoid training while/after this method is called.
    """

    checkpoints = sorted(model.checkpoint_mgr.checkpoints, key=lambda x: (len(x), x))
    checkpoints = checkpoints[-last_n:]

    sum_weights = [tf.zeros_like(w) for w in model.trainable_variables]

    if last_n > len(checkpoints):
        print(f'Warning: Could not average weights from {last_n} checkpoints, '
              f'as only {len(checkpoints)} checkpoints were found.')

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

    print(f'\nAveraged {len(list(model.trainable_variables))} weights from {len(checkpoints)} checkpoints')

    # TODO Make sure we don't accidentally save the averaged model
    del model.checkpoint_mgr
    # model.checkpoint_mgr = None

