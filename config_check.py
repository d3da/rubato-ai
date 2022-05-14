import registry

from typing import Dict, Optional, List, Any, Type
from typing_inspect import is_union_type
from warnings import warn


# TODO make this a class method
def param_info(param: registry.ConfParam) -> str:
    used_by = ', '.join([x.class_name for x in registry.CONFIG_REG_BY_NAME[param.name]])
    return (f'\tName:       \t{param.name}\n'
            f'\tUsed by:    \t{used_by}\n'
            f'\tType:       \t{param.conf_type}\n'
            f'\tDefault:    \t{param.default}\n'
            f'\tDescription:\t{param.description}\n')


class ConfParamException(Exception):
    def __init__(self, msg: str, param: registry.ConfParam):
        super().__init__(f'{msg}\n{param_info(param)}')

class ConfParamUnsetError(ConfParamException):
    def __init__(self, param: registry.ConfParam):
        super().__init__('Configuration parameter unset.', param)

class ConfParamTypeError(ConfParamException):
    def __init__(self, param: registry.ConfParam, value: Any):
        super().__init__(f'Configuration parameter has wrong type. (got {type(value)})', param)


def _check_param(param: registry.ConfParam, **config):
    if param.name not in config:
        raise ConfParamUnsetError(param)

    # Comment out this function call if the type checking causes problems
    _check_param_type(param, **config)


def _check_param_type(param: registry.ConfParam, **config):
    """
    Check whether a config parameter has the type defined in @register_param at runtime.
    This check currently uses an import (typing_inspect) and may cause problems with
    different python versions (tested on 3.8).
    """

    value = config[param.name]

    # Special case for Union[...] types, they cannot be checked normally
    # see https://bugs.python.org/issue44529
    if is_union_type(param.conf_type):
        _found_type = False
        for t in param.conf_type.__args__:
            if isinstance(value, t):
                _found_type = True
        if not _found_type:
            raise ConfParamTypeError(param, value)

    elif not isinstance(value, param.conf_type):
        raise ConfParamTypeError(param, value)


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
    """
    Recursively validate a configuration dict against the registry of parameters.
    (it ensures parameters are defined and have the right type)

    Checks all parameters registered for the class with name 'check_class',
    as well as other classes that 'check_class' links to.
    """
    if visited_classes is None:
        visited_classes = []
    if check_class in visited_classes:
        raise RuntimeError(f'Error: Class {check_class} already visited.')

    # Check parameters used by class
    if check_class in registry.CONFIG_REG_BY_CLASS_NAME:
        for param in registry.CONFIG_REG_BY_CLASS_NAME[check_class]:
            _check_param(param, **config)

    # Check parameters that define optional links and (optionally) check the linked classes
    if check_class in registry.CONFIG_REG_OPTIONAL_LINKS:
        for opt_param, opt_choices in registry.CONFIG_REG_OPTIONAL_LINKS[check_class].items():
            _check_option_param(opt_param, opt_choices, visited_classes, **config)

    # Check classes directly linked to
    if check_class in registry.CONFIG_REG_CLASS_LINKS:
        for created_class in registry.CONFIG_REG_CLASS_LINKS[check_class]:
            check_config(created_class, visited_classes, **config)


if __name__ == '__main__':
    from config import default_conf
    from train import ModelTrainer
    check_config('ModelTrainer', **default_conf)

