import os
import time

import tensorflow as tf

from .optimizer import Optimizer
from .callbacks import TrainCallback
from .input_loader import PerformanceInputLoader
from .midi_processor import MidiProcessor
from .registry import register_param, register_links, document_registrations, \
        PathLike, ConfDict, CONFIG_REGISTRY
from .exceptions import CheckpointIncompatibleError

from typing import Iterable, Optional


@document_registrations
@register_param('train_dir', PathLike,
                'Path for saving checkpoints, tensorboard logs and samples')
@register_param('kept_checkpoints', int,
                'Number of checkpoints to save in checkpoint directory')
@register_param('label_smoothing', float,
                'Amount of label smoothing regularization to apply to training examples')
@register_links({'Optimizer', 'TrainCallback', 'MidiProcessor'})
class BaseModel(tf.keras.Model):
    """
    Base class inherited by TransformerModel and RnnModel.
    This class can be considered abstract and is not instantiated directly.

    This class handles the following:
        - Setup the optimizer + loss
        - Run the train() loop
        - Keep persistent batch / epoch counters
        - Save checkpoints

    .. todo::
        - Move checkpoint compatibility logic to a different (Mixin) class
        - Instantiate the optimizer and loss in RubatoAI (like the input loader)
        - Abstract methods for sample_music etc.
    """
    _config_attr_prefix = '_config_attr_prefix'

    def __init__(self,
                 model_name: str,
                 input_loader: PerformanceInputLoader,
                 restore_checkpoint: bool,
                 config: ConfDict):
        super().__init__(name=model_name)
        self.input_loader = input_loader
        self.train_dir = config['train_dir']

        self.midi_processor = MidiProcessor(config)
        self.vocab_size = self.midi_processor.vocab_size

        self._batch_ctr = tf.Variable(0, trainable=False, dtype=tf.int64)
        self._epoch_ctr = tf.Variable(0, trainable=False, dtype=tf.int64)

        self.optimizer = Optimizer.create(step_counter=self._batch_ctr, config=config)

        self.loss = tf.losses.CategoricalCrossentropy(from_logits=True,
                                                      label_smoothing=config['label_smoothing'])
        self.compile(optimizer=self.optimizer, loss=self.loss,
                     metrics=['accuracy'])

        # Save all config items as class attributes, so they are saved to any checkpoints as well
        # Only objects trackable by tensorflow are saved to the checkpoint
        for k, v in config.items():
            self.__setattr__(self._config_attr_prefix+k, tf.Variable(v, trainable=False))

        # Setup the checkpoint manager
        checkpoint_dir = os.path.join(self.train_dir, 'checkpoints', model_name)
        checkpoint = tf.train.Checkpoint(model=self, optimizer=self.optimizer)
        self.checkpoint_mgr = tf.train.CheckpointManager(
            checkpoint,
            directory=checkpoint_dir,
            max_to_keep=config['kept_checkpoints']
        )

        print('\n') # print some newlines so we see where the tensorflow warnings end

        self._restored_checkpoint = False
        if restore_checkpoint:
            self._ckpt_restore = checkpoint.restore(self.checkpoint_mgr.latest_checkpoint)
            if self.batch_count != 0:
                self._restored_checkpoint = True
                print(f'Restored checkpoint (batch {self.batch_count}, epoch {self.epoch_count})\n')
            else:
                # We are assuming that this means we are creating a new model
                # TODO check whether a checkpoint exists for model_name?
                print('Initialized model (we\'re at batch zero)\n')

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

    def check_checkpoint_compatibility(self, config: ConfDict) -> None:
        """
        Check if the supplied config differs from the the loaded checkpoint's config.
        If any config dict values have been changed, error out.

        .. todo::
            - Error only for config params that are used (use with config_check check)
            - Accumulate errors and error out at the end (like validate_config does)
        """
        if not self._restored_checkpoint:
            return

        for param_name, value in config.items():
            old_value = getattr(self, self._config_attr_prefix + param_name).value()
            if old_value == value:
                continue

            if CONFIG_REGISTRY.breaks_checkpoint_compatibility(param_name):
                # Don't warn about unused ckpt objects on exit
                self._ckpt_restore.expect_partial()
                raise CheckpointIncompatibleError(param_name, old_value, value)

            else:
                print(f'Warning: \'{param_name}\' changed from {old_value} to {value}')
                # Update the attribute so the warning is shown only once
                self.__setattr__(self._config_attr_prefix + param_name,
                                 tf.Variable(value, trainable=False))

    def sample_music(self, sample_length: int, temperature: float,
                     num_samples: int, verbose: bool):
        """
        Abstract method to be implemented by subclasses
        .. todo::
            Primers, unify arguments between transformer/rnn models etc
        """
        raise NotImplementedError

    def save_samples(self, samples: Iterable, sample_directory: PathLike):
        """
        Save each sample from an iterable of model-generated sample to disk

        .. todo::
            - Call sample_music from here
            - Better path collision prevention
        """
        for i, seq in enumerate(samples):
            events = self.midi_processor.indices_to_events(seq)
            midi = self.midi_processor.events_to_midi(events)

            # TODO this is a terrible way to prevent overwriting lol
            for _ in range(9999):
                midi_path = os.path.join(sample_directory, f'{self.name}_{self.batch_count}_{i}.midi')
                if not os.path.exists(midi_path):
                    break
                i += 1
            else:
                raise FileExistsError

            midi.save(midi_path)



# if __name__ == '__main__':

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
