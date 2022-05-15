from .base_model import PerformanceModel
from .transformer import TransformerModel
from .performance_rnn import PerformanceRNNModel
from .input_loader import PerformanceInputLoader
from .registry import register_param, register_link_param, register_links
from .config_check import validate_config

from .config import default_conf

import tensorflow as tf

@register_link_param('model_type', {
    'transformer': 'TransformerModel',
    'rnn' : 'PerformanceRNNModel',
})
@register_param('batch_size', int, 'Batch size to use during training')
@register_param('sequence_length', int, '(Maximum) input sequence length')
@register_param('mixed_precision', bool, 'Enable mixed_float16 precision in session')
@register_links({'PerformanceInputLoader'})
class ModelTrainer:
    """
    'Start' class used to instantiate the input loader and a selected subclass of PerformanceModel,
    based on the supplied configuration dictionary.

    TODO think of a better name for this class (or change PerformanceModel.__name__)
    TODO move this to base_model or something?
    """

    def __init__(self, model_name: str, restore_checkpoint: bool, **config):
        validate_config(type(self).__name__, **config)

        self.mixed_precision = config['mixed_precision']
        if self.mixed_precision:
            print('>>>>> Enabling mixed precision floats')
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

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
        if config['model_type'] == 'rnn':
            return PerformanceRNNModel(model_name, input_loader, restore_checkpoint, **config)
        raise ValueError

    def train(self, epochs: int):
        self.model.train(epochs)


def main():
    trainer = ModelTrainer('ModelName', restore_checkpoint=False, **default_conf)
    trainer.train(10)

if __name__ == '__main__':
    main()
