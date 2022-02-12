from typing import Optional

import tensorflow as tf


class Optimizer:
    @staticmethod
    def create_adam_optimizer(step_counter: Optional[tf.Variable], **config):
        adam_kwargs = dict()  # arguments for tf.keras.optimizers.Adam

        learning_rate = config['learning_rate']
        if learning_rate is not None:
            # Default adam, no warmup schedule
            adam_kwargs['learning_rate'] = learning_rate
        else:
            # Noam learning rate schedule with warmup
            # Vaswani et al. (2017)
            assert config.get('warmup_steps') > 0
            assert step_counter is not None
            assert config.get('embed_dim') is not None

            def noam_lr_schedule():
                step = tf.cast(step_counter.value(), tf.float32)
                min_part = tf.math.minimum(step**(-0.5), step*config['warmup_steps']**(-1.5))
                return config['embed_dim']**(-0.5) * min_part

            adam_kwargs['learning_rate'] = noam_lr_schedule

        adam_kwargs['beta_1'] = config.get('adam_beta1')
        adam_kwargs['beta_2'] = config.get('adam_beta2')
        adam_kwargs['epsilon'] = config.get('adam_eps')

        return tf.keras.optimizers.Adam(**adam_kwargs)

