import registry

from typing import Dict, Optional, List
from warnings import warn


class ConfParamException(Exception):
    def __init__(self, msg: str, param: registry.ConfParam):
        super().__init__(f'{msg}\n{param_info(param)}')

def param_info(param: registry.ConfParam) -> str:
    used_by = ', '.join([x.class_name for x in registry.CONFIG_REG_BY_NAME[param.name]])
    return (f'\tName:       \t{param.name}\n'
            f'\tUsed by:    \t{used_by}\n'
            f'\tType:       \t{param.conf_type}\n'
            f'\tDefault:    \t{param.default}\n'
            f'\tDescription:\t{param.description}\n')


class ConfParamUnsetError(ConfParamException):
    def __init__(self, param: registry.ConfParam):
        super().__init__('Configuration parameter unset.', param)


def _check_param(param: registry.ConfParam, **config):
    """
    TODO check type / value, possibly set default
    """
    if param.name not in config:
        raise ConfParamUnsetError(param)

def _check_option_param(option_param: str,
                        option_choices: Dict[str, str],
                        visited_clases: List[str],
                        **config):
    if option_param not in config:
        print(f'Warning: Config parameter \'{option_param}\' is unset.')
        warn(f'Config parameter \'{option_param}\' is unset.\n'
             f'\tOptions: {list(option_choices.keys())}')
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

    if check_class in registry.CONFIG_REG_OPTIONAL_LINKS:
        for opt_param, opt_choices in registry.CONFIG_REG_OPTIONAL_LINKS[check_class].items():
            _check_option_param(opt_param, opt_choices, visited_classes, **config)

    if check_class in registry.CONFIG_REG_CLASS_LINKS:
        for created_class in registry.CONFIG_REG_CLASS_LINKS[check_class]:
            check_config(created_class, visited_classes, **config)



