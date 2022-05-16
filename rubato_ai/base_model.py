"""

"""
import os
import time

import tensorflow as tf

from .optimizer import Optimizer
from .callbacks import TrainCallback
from .input_loader import PerformanceInputLoader

from .registry import register_param, register_links, PathLike, ConfDict


@register_param('train_dir', PathLike,
                'Path for saving checkpoints, tensorboard logs and samples')
@register_param('kept_checkpoints', int,
                'Number of checkpoints to save in checkpoint directory')
@register_param('label_smoothing', float,
                'Amount of label smoothing regularization to apply to training examples')
@register_links({'Optimizer', 'TrainCallback'})
class BaseModel(tf.keras.Model):
    """
    Base class inherited by TransformerModel and RnnModel.
    This class can be considered abstract and is not instantiated directly.

    This class handles the following:
        - Setup the optimizer
        - Run the train() loop
        - Keep persistent batch / epoch counters
        - Save checkpoints
    """
    def __init__(self,
                 model_name: str,
                 input_loader: PerformanceInputLoader,
                 restore_checkpoint: bool,
                 config: ConfDict):
        super().__init__(name=model_name)
        self.input_loader = input_loader
        self.train_dir = config['train_dir']

        self._batch_ctr = tf.Variable(0, trainable=False, dtype=tf.int64)
        self._epoch_ctr = tf.Variable(0, trainable=False, dtype=tf.int64)

        self.optimizer = Optimizer.create(step_counter=self._batch_ctr, config=config)

        self.loss = tf.losses.CategoricalCrossentropy(from_logits=True,
                                                      label_smoothing=config['label_smoothing'])
        self.compile(optimizer=self.optimizer, loss=self.loss,
                     metrics=['accuracy'])

        checkpoint_dir = os.path.join(self.train_dir, 'checkpoints', model_name)
        checkpoint = tf.train.Checkpoint(model=self, optimizer=self.optimizer)
        self.checkpoint_mgr = tf.train.CheckpointManager(
            checkpoint,
            directory=checkpoint_dir,
            max_to_keep=config['kept_checkpoints']
        )
        if restore_checkpoint:
            checkpoint.restore(self.checkpoint_mgr.latest_checkpoint)
            if self.batch_count != 0:
                print(f'Restored checkpoint (batch {self.batch_count}, epoch {self.epoch_count})')
            else:
                print('Initialized model (we\'re at batch zero)')

        self.callbacks = [TrainCallback(config)]
        self.load_time = time.localtime()

    @property
    def batch_count(self):
        return self._batch_ctr.value().numpy()

    @property
    def epoch_count(self):
        return self._epoch_ctr.value().numpy()

    def increment_batch(self):
        return self._batch_ctr.assign_add(1).value().numpy()

    def increment_epoch(self):
        return self._epoch_ctr.assign_add(1).value().numpy()

    def train(self, epochs: int) -> None:
        """
        Note:
            Instead of simply calling ``self.fit()`` with epoch=epochs,
            we call ``fit(epochs=1)`` once for each training epoch.
            This is because the train dataset varies in length between epochs,
            which ``fit()`` cannot handle normally.
            The drawback is that we don't get an epoch ETA timer.
        """
        for e in range(epochs):
            self.fit(self.input_loader.dataset, epochs=1,
                     callbacks=self.callbacks)
            print(f'Finished training epoch {e+1}/{epochs}.')




if __name__ == '__main__':
    exit()

    # # Simon & Oore (2018)
    # inner_model = RnnModel(
    #     vocab_size=input_loader.vocab_size,
    #     rnn_units=512,
    #     dropout=0.0
    # )

    # # Vaswani 2017
    # inner_model = TransformerModel(
    #     vocab_size=input_loader.vocab_size,
    #     sequence_length=512,
    #     num_layers=6,
    #     drop_rate=0.1,
    #     embed_dim=512,
    #     attn_heads = 8,
    #     ff_dim = 2048,
    #     attn_dim=None
    # )

    # Huang 2018 (Baseline transformer)
    # inner_model = TransformerModel(
    #     vocab_size=input_loader.vocab_size,
    #     sequence_length=2048,
    #     num_layers=8,
    #     drop_rate=0.2,
    #     embed_dim=384,
    #     attn_heads=8,
    #     ff_dim=1024,
    #     attn_dim=512
    # )
    #
    # model = BaseModel(
    #     inner_model,
    #     input_loader,
    #     'outer_model',  #todo don't allow 2 different model types wiith same name
    #     PROJECT_DIR,
    #     restore_checkpoint=True,
    #
    #     # # Simon & Oore (2018)
    #     # learning_rate=1e-3,
    #
    #     # Vaswani et al. (2017)
    #     learning_rate=None,
    #     adam_beta1=0.9,
    #     adam_beta2=0.98,
    #     adam_eps=1e-9,
    #     warmup_steps=4000,
    #     # embed_dimension=512,
    #     label_smoothing=0.1,
    #
    #     # Huang et al. (2018)
    #     embed_dimension=384,
    # )
    #
    # model.__call__(tf.zeros((1, 2048), dtype=tf.int32))
    # model.summary()
    # model.train(1)
    # sys.exit()
