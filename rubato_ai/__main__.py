"""
Command-line interface to RubatoAI.

.. todo::
    - number of epochs to train for
    - Sampling (maybe using avg_checkpoints)
    - Define action using sub_commands
          `<https://docs.python.org/dev/library/argparse.html#sub-commands>`_
    - move model-name to config.py
    - Flag to skip checkpoint compatibility check
    - Help strings for each argument
    - Should we be able to --sample with --no-restore-checkpoint?
    - Should we be able to --check with --skip-config-check?
"""
import argparse

from .rubato_ai import RubatoAI
from .config_check import validate_config

import config

parser = argparse.ArgumentParser(description='Train or sample a RubatoAI model.')

parser.add_argument('--model-name', required=True, type=str,
                    help='Name of the model instance. Checkpoints, logs and samples will be saved under this name.')

parser.add_argument('--config', default='default_conf', nargs='?',
                    help='Configuration from config.py to use as model configuration. Defaults to \'default_conf\'.')

parser.add_argument('--no-restore-checkpoint', action='store_false', dest='restore_checkpoint',
                    help='Do not load the checkpoint from disk. Checkpoints will still be saved.')

parser.add_argument('--skip-config-check', action='store_true', dest='skip_config_check',
                    help=None)

parser.add_argument('action', choices=['train', 'check', 'sample'], default='train', nargs='?',
                    help=None)

# parser.print_help()
args = parser.parse_args()
# print()
print(args)


config_dict = getattr(config, args.config)  # raises AttributeError on failure
assert isinstance(config_dict, dict), 'Configuration object supplied with --config must be a dict'

if args.action == 'train':
    rubato = RubatoAI(args.model_name, args.restore_checkpoint, config=config_dict,
                      skip_config_check=args.skip_config_check, train_mode=True)

    exit(rubato.train(epochs=10))

elif args.action == 'check':
    exit(validate_config('RubatoAI', config=config_dict))

elif args.action == 'sample':
    rubato = RubatoAI(args.model_name, args.restore_checkpoint, config=config_dict,
                      skip_config_check=args.skip_config_check, train_mode=False)
    raise NotImplementedError('No support for sampling without training yet... #TODO')

raise ValueError
