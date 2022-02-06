from typing import Optional

import tensorflow as tf


class Optimizer:
    """
    TODO document this mess
    """

    @staticmethod
    def create_adam_optimizer(
            learning_rate: Optional[float],
            adam_beta1: Optional[float],
            adam_beta2: Optional[float],
            adam_eps: Optional[float],
            warmup_steps: Optional[int],
            step_counter: Optional[tf.Variable],
            embed_dimension: Optional[int],
    ):
        kwargs = dict()
        if learning_rate is not None:
            # Default adam, no warmup schedule
            kwargs['learning_rate'] = learning_rate
        else:
            # Noam learning rate schedule with warmup
            # Vaswani et al. (2017)
            assert warmup_steps > 0
            assert step_counter is not None
            assert embed_dimension is not None

            def noam_lr_schedule():
                step = tf.cast(step_counter.value(), tf.float32)
                min_part = tf.math.minimum(step**(-0.5), step*warmup_steps**(-1.5))
                return embed_dimension**(-0.5) * min_part

            kwargs['learning_rate'] = noam_lr_schedule

        if adam_beta1 is not None:
            kwargs['beta_1'] = adam_beta1
        if adam_beta2 is not None:
            kwargs['beta_2'] = adam_beta2
        if adam_eps is not None:
            kwargs['epsilon'] = adam_eps

        return tf.keras.optimizers.Adam(**kwargs)

