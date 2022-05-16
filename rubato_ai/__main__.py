"""
Command-line interface to RubatoAI.

.. todo:
    - Select configuration based on setting here
    - number of epochs to train for
    - Sampling (maybe using avg_checkpoints)
    - Define action using sub_commands
          `<https://docs.python.org/dev/library/argparse.html#sub-commands>`_
"""
import argparse

from .rubato_ai import RubatoAI
from .config_check import validate_config
from .config import default_conf

parser = argparse.ArgumentParser(description='Train or sample a RubatoAI model.')

parser.add_argument('--model-name', required=True, type=str)
parser.add_argument('--config', default='default_conf', nargs='?')
parser.add_argument('--no-restore-checkpoint', action='store_false', dest='restore_checkpoint')
parser.add_argument('action', choices=['train', 'check', 'sample'], default='train', nargs='?')

parser.print_help()
args = parser.parse_args()
print()
print(args)


# TODO set this dynamically
config = default_conf


if args.action == 'train':
    rubato = RubatoAI(args.model_name, args.restore_checkpoint, **default_conf)
    exit(rubato.train(epochs=10))

elif args.action == 'check':
    exit(validate_config('RubatoAI', **default_conf))

elif args.action == 'sample':
    raise NotImplementedError('No support for sampling without training yet... #TODO')

raise ValueError
