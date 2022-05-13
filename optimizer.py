from typing import Optional, Union, Callable

from registry import register_param, register_optional_links, register_links

import tensorflow as tf


@register_optional_links('learning_rate_schedule', {
    'standard' : 'StandardLearningRateSchedule',
    'noam' : 'NoamLearningRateSchedule',
})
@register_links({'AdamOptimizer'})
class Optimizer:
    """Class to build an Adam optimizer using config parameters."""

    @staticmethod
    def create(step_counter: Optional[tf.Variable], **config) -> 'AdamOptimizer':
        """
        Build an adam optimizer instance given the configuration.

        Needs a reference to the model's step counter to calculate the learning rate,
            when using the 'noam' schedule.
        """
        learning_rate = Optimizer._learning_rate_schedule_from_config(step_counter, **config)
        return AdamOptimizer(learning_rate, **config)

    @staticmethod
    def _learning_rate_schedule_from_config(step_counter: Optional[tf.Variable], **config) -> Union[int, Callable]:
        sched = config.get('learning_rate_schedule')

        if sched == 'standard':
            return config['learning_rate']

        elif sched == 'noam':
            assert config.get('warmup_steps') is not None
            assert config.get('embed_dim') is not None
            assert step_counter is not None
            return NoamLearningRateSchedule(step_counter, **config)

        raise ValueError('config.learning_rate_schedule must be either \'standard\' or \'noam\'')


@register_param('warmup_steps', 'int', 4000, 'Warmup steps for Noam schedule')
@register_param('embed_dim', 'int', 512, 'Model hidden dimension size')
class NoamLearningRateSchedule:
    """Noam Shazeer's learning rate schedule as proposed in Vaswani et al. 2017."""
    def __init__(self, step_counter: tf.Variable, **config):
        self._step_counter = step_counter
        self._warmup_steps = config['warmup_steps']
        self._embed_dim = config['embed_dim']

    def __call__(self, *args, **kwargs):
        step = tf.cast(self._step_counter.value(), tf.float32)
        min_part = tf.math.minimum(step ** (-0.5), step * self._warmup_steps ** (-1.5))
        return self._embed_dim ** (-0.5) * min_part


@register_param('learning_rate', 'float', 1e-3, 'Adam optimizer learning rate')
class StandardLearningRateSchedule:
    """Dummy class used to register the optional learning rate hyperparameter"""
    pass


@register_param('adam_beta1', 'float', 0.9, 'Beta1 parameter of Adam optimizer')
@register_param('adam_beta2', 'float', 0.98, 'Beta2 parameter of Adam optimizer')
@register_param('adam_eps', 'float', 1e-9, 'Epsilon parameter of Adam optimizer')
class AdamOptimizer(tf.keras.optimizers.Adam):
    """Wrapper for tf.keras.optimizers.Adam that registers config paramters"""
    def __init__(self, learning_rate: Union[int, Callable], **config):
        super().__init__(beta_1=config.get('adam_beta1'),
                         beta_2=config.get('adam_beta1'),
                         epsilon=config.get('adam_eps'),
                         learning_rate=learning_rate)
