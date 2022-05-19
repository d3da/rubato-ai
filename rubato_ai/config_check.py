from .registry import CONFIG_REGISTRY, ConfDict, ConfParam, LinkParam
from .exceptions import ConfigException, ConfParamException, ConfParamUnsetError, \
        ConfParamTypeError, LinkParamException, LinkParamUnsetError, LinkParamWrongValueError

from typing import Optional, List
from typing_inspect import is_union_type

import sys


def check_unused_params(config: ConfDict):
    """
    Warn the user about config parameters in the config dictionary that
    are not registered anywhere in the imported modules.
    """
    for param_name in config:
        # Check if parameter name is found in conf params
        if param_name not in CONFIG_REGISTRY.conf_params_by_name:

            # Check if param name is used by any registered link params
            found_usage = False
            for params_by_class in CONFIG_REGISTRY.link_params.values():
                for link_param in params_by_class:
                    if param_name == link_param.choice_param:
                        found_usage = True

            if not found_usage:
                print(f'Warning: Parameter \'{param_name}\' is unknown', file=sys.stderr)


def _check_param(param: ConfParam, config: ConfDict):
    if param.name not in config:
        raise ConfParamUnsetError(param)

    # Comment out this function call if the type checking causes problems
    _check_param_type(param, config)


def _check_param_type(param: ConfParam, config: ConfDict):
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


def _check_link_param(link_param: LinkParam,
                      config: ConfDict,
                      _visited_clases: List[str]) -> int:
    if link_param.choice_param not in config:
        raise LinkParamUnsetError(link_param)

    choice = config[link_param.choice_param]
    if choice not in link_param.choice_options:
        raise LinkParamWrongValueError(link_param, choice)
    
    return check_config(link_param.choice_options[choice], config, _visited_clases)


def check_config(check_class: str,
                 config: ConfDict,
                 _visited_classes: Optional[List[str]] = None) -> int:
    """
    Recursively validate a configuration dict against the registry of parameters.
    (it ensures parameters are defined and have the right type)

    Checks all parameters registered for the class with name 'check_class',
    as well as other classes that 'check_class' links to.

    Returns the number of errors found in configuration
    """
    _visited_classes = [] if _visited_classes is None else _visited_classes[:]

    # Avoid getting stuck in a loop
    if check_class in _visited_classes:
        raise RuntimeError(f'Class {check_class} already visited. Is there a circular link?')
    _visited_classes.append(check_class)

    # Check if the class has anything registered
    if check_class not in CONFIG_REGISTRY.conf_params_by_class_name and \
            check_class not in CONFIG_REGISTRY.link_params and \
            check_class not in CONFIG_REGISTRY.class_links:
        linked_from = ''
        if _visited_classes is not None and len(_visited_classes) > 2:
            linked_from = f'linked from {_visited_classes[-2]} '
        raise RuntimeError(f'Class {check_class} {linked_from}'
                            'was not registered.')

    num_errors = 0

    # Check parameters used by class
    if check_class in CONFIG_REGISTRY.conf_params_by_class_name:
        for param in CONFIG_REGISTRY.conf_params_by_class_name[check_class]:
            try:
                _check_param(param, config)
            except ConfParamException as e:
                num_errors += 1
                print(e, file=sys.stderr)

    # Check link parameters and check the linked classes (based on the set value)
    if check_class in CONFIG_REGISTRY.link_params:
        for link_param in CONFIG_REGISTRY.link_params[check_class]:
            try:
                num_errors += _check_link_param(link_param, config, _visited_classes)
            except LinkParamException as e:
                num_errors += 1
                print(e, file=sys.stderr)

    # Check classes directly linked to
    if check_class in CONFIG_REGISTRY.class_links:
        for created_class in CONFIG_REGISTRY.class_links[check_class]:
            num_errors += check_config(created_class, config, _visited_classes)

    return num_errors

def validate_config(check_class: str, config: ConfDict):
    """
    Run :meth:`check_config` but raise an exception at the end if errors were encountered,
    and warn about unused parameters beforehand (see :meth:`check_unused_params`)

    :param check_class: Name of class to start validation at
    :param conf: Configuration dictionary
    """
    check_unused_params(config)
    num_errors = check_config(check_class, config)
    if num_errors > 0:
        raise ConfigException(f'Found {num_errors} errors in configuration')
