import registry

from typing import Dict, Optional, List


def _check_param(param: registry.ConfParam, **config):
    """
    TODO check type / value, possibly set default
    """
    if param.name not in config:
        print(f'Warning: Config parameter \'{param.name}\' is unset.')
        raise RuntimeWarning

def _check_option_param(option_param: str,
                        option_choices: Dict[str, str],
                        visited_clases: List[str],
                        **config):
    if option_param not in config:
        print(f'Warning: Config parameter \'{option_param}\' is unset.')
        raise RuntimeWarning
    # TODO set default

    choice = config[option_param]
    if choice not in option_choices:
        print(f'Error: Config parameter \'{option_param}\' must be set to one of '
              f'\t{list(option_choices.keys())}')
        raise RuntimeError
    
    check_config(option_choices[choice], visited_clases, **config)


def check_config(check_class: str,
                 visited_classes: Optional[List[str]] = None,
                 **config):
    if visited_classes is None:
        visited_classes = []
    if check_class in visited_classes:
        print(f'Error: Class {check_class} already visited.')
        raise RuntimeError

    if check_class in registry.CONFIG_REG_BY_CLASS_NAME:
        for param in registry.CONFIG_REG_BY_CLASS_NAME[check_class]:
            _check_param(param, **config)

    if check_class in registry.CONFIG_REG_CLASS_CREATES:
        for created_class in registry.CONFIG_REG_CLASS_CREATES[check_class]:
            check_config(created_class, visited_classes, **config)

    if check_class in registry.CONFIG_REG_OPTIONAL_CREATES:
        for option_param, option_choices in registry.CONFIG_REG_OPTIONAL_CREATES[check_class].items():
            _check_option_param(option_param, option_choices, visited_classes, **config)


