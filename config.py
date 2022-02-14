#!/usr/bin/env python3
import os
from typing import Dict, Any

import tensorflow as tf

from base_model import PerformanceModel
from input_loader import PerformanceInputLoader
from performance_rnn import PerformanceRNNModel
from transformer import TransformerModel

PROJECT_DIR = os.path.dirname(__file__)

default_conf = {

    # Input loader settings
    'sequence_length': 2048,
    'batch_size': 2,
    'augmentation': 'aug-',
    'time_shift_ms': None,  # TODO
    'min_stride': 512,
    'max_stride': 1024,

    # dataset generator settings
    'shuffle_buffer_size': 8096,
    'queue_size': 32,  # No impact on model performance
    'num_threads': 4,  # No impact on model performance (may impact throughput)

    # TODO Performance RNN / Vaswani hparams

    # Model settings: (Transformer)
    'num_layers': 8,
    'drop_rate': 0.2,
    'embed_dim': 512,
    'attn_dim': 384,
    'attn_heads': 8,
    'ff_dim': 1024,
    'attn_type': 'relative',
    'max_relative_pos': 1024,  # TODO subparam?


    # Optimizer settings
    'learning_rate': None,
    'warmup_steps': 4000,
    'adam_beta1': 0.9,
    'adam_beta2': 0.98,
    'adam_eps': 1e-9,
    'label_smoothing': 0.1,


    # Train loop settings
    'tensorboard_update_freq': 50,
    'sample_midi_freq': 250,
    'sample_midi_length': 512,
    'validation_freq': 1000,
    'validation_batches': 25,
    'save_checkpoint_freq': 500,
    'kept_checkpoints': 50,

    'train_dir': PROJECT_DIR,
    # 'dataset_dir': 'data/maestro-v3.0.0',
    # 'dataset_csv': 'maestro-v3.0.0.csv',
}


def load_model_from_config(config: Dict[str, Any]) -> PerformanceModel:
    dataset_base = os.path.join(config['train_dir'], 'data/maestro-v3.0.0')
    dataset_csv = os.path.join(dataset_base, 'maestro-v3.0.0.csv')

    input_loader = PerformanceInputLoader(
        dataset_base,
        dataset_csv,
        sequence_length=config['sequence_length'],
        min_stride=config['min_stride'],
        max_stride=config['max_stride'],
        batch_size=config['batch_size'],
        augmentation=config['augmentation'],
        shuffle_buffer_size=config['shuffle_buffer_size'],
        queue_size=config['queue_size'],
        num_threads=config['num_threads'],
    )

    model = TransformerModel(
        input_loader,
        'MyModel',
        restore_checkpoint=False,
        **config
    )
    # model = PerformanceRNNModel(
    #     'PerfRNN',
    #     input_loader,
    #     False,
    #     vocab_size=input_loader.vocab_size,
    #     rnn_units=512,
    #     dropout=0.0,
    #     **config
    # )

    model.__call__(tf.zeros((config['batch_size'],
                             config['sequence_length']), dtype=tf.int32))
    model.summary()
    return model


if __name__ == '__main__':
    model = load_model_from_config(default_conf)
    model.train(1)
