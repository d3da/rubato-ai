import os

from .base_model import BaseModel
from .transformer import TransformerModel
from .rnn import RnnModel
from .input_loader import PerformanceInputLoader
from .registry import register_param, register_link_param, register_links, \
        document_registrations, ConfDict
from .config_check import validate_config

import tensorflow as tf

@document_registrations
@register_link_param('model_type', {
    'transformer': 'TransformerModel',
    'rnn' : 'RnnModel',
})
@register_param('batch_size', int, 'Batch size to use during training')
@register_param('sequence_length', int, '(Maximum) input sequence length')
@register_param('mixed_precision', bool,
                'Enable mixed_float16 precision in session',
                breaks_compatibility=True)
class RubatoAI:
    """
    'Start' class used to instantiate the input loader and a selected subclass of BaseModel,
    based on the supplied configuration dictionary.

    .. todo::
        - rename validate_config to check_config or sth
    """

    def __init__(self, restore_checkpoint: bool, config: ConfDict,
                 skip_config_check: bool, train_mode: bool):
        if not skip_config_check:
            validate_config(type(self).__name__, config=config)

        self.mixed_precision = config['mixed_precision']
        if self.mixed_precision:
            print('>>>>> Enabling mixed precision floats')
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

        self.model = self._model_from_config(train_mode,
                                             restore_checkpoint,
                                             config)

        self.model.check_checkpoint_compatibility(config)

        self.model.__call__(tf.zeros((config['batch_size'],
                                      config['sequence_length']), dtype=tf.int32))
        self.model.summary()

    @staticmethod
    def _model_from_config(train_mode: bool,
                           restore_checkpoint: bool,
                           config: ConfDict) -> BaseModel:
        if config['model_type'] == 'transformer':
            return TransformerModel(train_mode, restore_checkpoint, config)
        if config['model_type'] == 'rnn':
            return RnnModel(train_mode, restore_checkpoint, config)
        raise ValueError

    def train(self, epochs: int):
        self.model.train(epochs)

    def sample(self, config: ConfDict, sample_length: int = 2048, temperature: float = 0.9,
               num_samples: int = 2):

        music = self.model.sample_music(sample_length=sample_length, temperature=temperature,
                                        num_samples=num_samples, verbose=True)
        sample_dir = 'samples'
        sample_subdir = os.path.join(sample_dir, self.model.name)
        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)
        if not os.path.exists(sample_subdir):
            os.mkdir(sample_subdir)
        self.model.save_samples(music, sample_subdir)
