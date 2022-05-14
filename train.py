#!/usr/bin/env python3
from base_model import PerformanceModel
from transformer import TransformerModel
from input_loader import PerformanceInputLoader
from registry import register_param, register_link_parameter, register_links
from config_check import check_config

from config import default_conf

import tensorflow as tf

@register_link_parameter('model_type', {
    'transformer': 'TransformerModel',
})
@register_param('batch_size', int, 'Batch size to use during training')
@register_param('sequence_length', int, '(Maximum) input sequence length')
@register_links({'PerformanceInputLoader'})
class ModelTrainer:
    """
    'Start' class used to instantiate the input loader and a selected subclass of PerformanceModel,
    based on the supplied configuration dictionary.

    TODO think of a better name for this class (or change PerformanceModel.__name__)
    """

    def __init__(self, model_name: str, restore_checkpoint: bool, **config):
        check_config(type(self).__name__, **config)

        self.input_loader = PerformanceInputLoader(**config)
        self.model = self._model_from_config(model_name,
                                             self.input_loader,
                                             restore_checkpoint,
                                             **config)

        self.model.__call__(tf.zeros((config['batch_size'],
                                      config['sequence_length']), dtype=tf.int32))
        self.model.summary()

    @staticmethod
    def _model_from_config(model_name: str,
                           input_loader: PerformanceInputLoader,
                           restore_checkpoint: bool,
                           **config) -> PerformanceModel:
        if config['model_type'] == 'transformer':
            return TransformerModel(model_name, input_loader, restore_checkpoint, **config)

        # TODO prnn

        raise ValueError

    def train(self, epochs: int):
        self.model.train(epochs)


if __name__ == '__main__':
    trainer = ModelTrainer('ModelName', restore_checkpoint=False, **default_conf)
    trainer.train(10)
