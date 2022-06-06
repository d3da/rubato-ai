"""
Command-line interface to RubatoAI.

.. todo::
    - number of epochs to train for
    - Flag to skip checkpoint compatibility check
    - Help strings for each argument
    - Should we be able to --sample with --no-restore-checkpoint?
    - Should we be able to --check with --skip-config-check?
    - Put subcommand help messages in main help
"""
import argparse

from .rubato_ai import RubatoAI
from .config_check import validate_config
from .avg_checkpoints import load_averaged_weights

import config

parser = argparse.ArgumentParser(description='Train or sample a RubatoAI model.')

# Global flags
parser.add_argument('--config', '-c', default='default_conf', nargs='?',
                    help='Configuration from config.py to use as model configuration. Defaults to \'default_conf\'.')

parser.add_argument('--no-restore-checkpoint', action='store_false', dest='restore_checkpoint',
                    help='Do not load the checkpoint from disk. Checkpoints will still be saved.')

parser.add_argument('--skip-config-check', action='store_true', dest='skip_config_check',
                    help='Do not validate the configuration dictionary before loading the model.')

# Subcommands
action_parser = parser.add_subparsers(help=None)
parser.set_defaults(action='train')

# Training mode
train_parser = action_parser.add_parser('train', help='Train a model (default)')
train_parser.set_defaults(action='train')

# Sampling mode
sample_parser = action_parser.add_parser('sample', help='Sample MIDI files from a model')
sample_parser.set_defaults(action='sample')
sample_parser.add_argument('--average-checkpoints', '-A', dest='average_checkpoints', type=int, 
                           help='If supplied, number of checkpoint weights averaged.')

# Check-only mode
check_parser = action_parser.add_parser('check', help='Validate the configuration then exit')
check_parser.set_defaults(action='check')

args = parser.parse_args()
parser.print_help()
print(args)

config_dict = getattr(config, args.config)  # raises AttributeError on failure
assert isinstance(config_dict, dict), 'Configuration object supplied with --config must be a dict'

if args.action == 'train':
    rubato = RubatoAI(args.restore_checkpoint, config=config_dict,
                      skip_config_check=args.skip_config_check, train_mode=True)

    exit(rubato.train(epochs=10))

elif args.action == 'check':
    exit(validate_config('RubatoAI', config=config_dict))

elif args.action == 'sample':

    rubato = RubatoAI(args.restore_checkpoint, config=config_dict,
                      skip_config_check=args.skip_config_check, train_mode=False)

    if args.average_checkpoints is not None:
        load_averaged_weights(rubato.model, last_n=args.average_checkpoints)

    exit(rubato.sample(config=config_dict))

raise ValueError
