#!/usr/bin/env python3
import os

PROJECT_DIR = os.path.dirname(__file__)

default_conf = {

    # Input loader settings
    'sequence_length': 2048,
    'batch_size': 2,
    'augmentation': 'aug-',
    'min_stride': 512,
    'max_stride': 1024,

    # Midi processor settings
    'time_granularity': 100,
    'piece_start': True,
    'piece_end': True,

    # dataset generator settings
    'shuffle_buffer_size': 8096,
    'queue_size': 32,  # No impact on model performance
    'num_threads': 4,  # No impact on model performance (may impact throughput)

    # TODO Performance RNN / Vaswani hparams

    # Model settings: (Transformer)
    'model_type': 'transformer',
    'num_layers': 8,
    'drop_rate': 0.2,
    'layernorm_eps': 1e-6,
    'embed_dim': 512,
    'attn_dim': 384,
    'attn_heads': 8,
    'ff_dim': 1024,
    'attn_type': 'relative',
    'max_relative_pos': 1024,  # TODO subparam?


    # Optimizer settings + label smoothing
    # 'learning_rate_schedule': 'standard',
    # 'learning_rate': 1e-3,
    'learning_rate_schedule': 'noam',
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
    'dataset_dir': os.path.join(PROJECT_DIR, 'data/maestro-v3.0.0'),
    'dataset_csv': 'maestro-v3.0.0.csv',  # relative to dataset_dir
}


