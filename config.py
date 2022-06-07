import os

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))

_path_config = {
    'train_dir': PROJECT_DIR,
    'dataset_dir': os.path.join(PROJECT_DIR, 'data/maestro-v3.0.0'),
    'dataset_csv': 'maestro-v3.0.0.csv',  # relative to dataset_dir
}

_data_config = {
    # Input loader
    'sequence_length': 2048,
    'augmentation': 'aug-',
    'min_stride': 1024,  # ?
    'max_stride': 2048,  # ?
    # Midi processor
    'time_granularity': 100,
    'piece_start': True,  # ?
    'piece_end': True,  # ?
    'max_silence': 8.0, # not in paper
    # dataset generator
    'shuffle_buffer_size': 8096,
    'queue_size': 16,  # No impact on model performance
    'num_threads': 14,  # TODO 'automatic' setting
}

_train_loop_config = {
    'tensorboard_update_freq': 25,
    'sample_midi_freq': 500,
    'sample_midi_length': 512,
    'validation_freq': 2500,
    'validation_batches': 250,
    'save_checkpoint_freq': 500,
    'kept_checkpoints': 5,
}

huang2018_relative_global_attention = {
    **_path_config,
    **_train_loop_config,
    **_data_config,

    'model_name': 'Huang2018_RelativeGlobalAttention',
    'mixed_precision': True,  # Should we use this?
    'batch_size': 2,

    'model_type': 'transformer',
    'num_layers': 8,
    'drop_rate': 0.2,
    'layernorm_eps': 1e-6,
    'embed_dim': 512,
    'attn_dim': 384,
    'attn_heads': 8,
    'ff_dim': 1024,
    'attn_type': 'relative',
    'max_relative_pos': 1024,
    'learning_rate_schedule': 'noam',
    'warmup_steps': 4000,
    'adam_beta1': 0.9,
    'adam_beta2': 0.98,
    'adam_eps': 1e-9,
    'label_smoothing': 0.1,
}
"""
Huang et al. 2018 Music Transformer with relative global attention.

Note:
    Parameters are actually taken from the 2019 ICLR conference version of the paper.
"""

default_conf = huang2018_relative_global_attention
"""Config to use if not overridden via command-line argument"""

# Model settings (RNN) TODO
# 'mixed_precision': False,
# 'model_type': 'rnn',
# 'drop_rate': 0.,
# 'rnn_units': 512,
# 'learning_rate_schedule': 'standard',
# 'learning_rate': 1e-3,
# 'adam_beta1': 0.9,
# 'adam_beta2': 0.98,
# 'adam_eps': 1e-9,
# 'label_smoothing': 0.0,
