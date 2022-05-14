"""
Keep a global registry of hyperparameters used.

Register a parameter directly accessed in a class by decorating the class with @register_param.
If a class creates other classes (having their own parameters) in its __init__ method, decorate the class with @register_creates.
"""
import os
from typing import Dict, Set, Union, Type

PathLike = Union[str, bytes, os.PathLike]


# All config parameters, accessed by parameter name
REG_CONF_PARAMS_BY_NAME: Dict[str, Set['ConfParam']] = {}

# All config parameters, accessed by class name
REG_CONF_PARAMS_BY_CLASS_NAME: Dict[str, Set['ConfParam']] = {}

# Set of classes created by key class during initialization, including possibly a superclass
REG_CLASS_LINKS: Dict[str, Set[str]] = {}

# Link parameters accesses by class name
REG_LINK_PARAMS: Dict[str, Set['LinkParam']] = {}


class ConfParam:
    def __init__(self,
                 class_name: str,
                 name: str,
                 conf_type: Type,
                 description: str):
        self.class_name = class_name
        self.name = name
        self.conf_type = conf_type
        self.description = description

    def __str__(self):
        used_by = ', '.join([x.class_name for x in REG_CONF_PARAMS_BY_NAME[self.name]])
        return (f'\tName:       \t{self.name}\n'
                f'\tUsed by:    \t{used_by}\n'
                f'\tType:       \t{self.conf_type}\n'
                f'\tDescription:\t{self.description}')


def register_param(name: str,
                   conf_type: Type,
                   description: str = 'No description provided'):
    """
    Register a hyperparameter to the parameter registry.
    Use as class decorator to define the config parameters used by that class.

    TODO handle two classes using same parameter
    TODO change config value to default if not present (warn the user)
    """

    def _wrap_class(cls):
        class_name = cls.__name__
        # print(f'{class_name}: Registering config parameter \'{name}\'')
        # print(f'\tType: {conf_type}\n\tDescription: {description}\n\tDefault: {default}\n')

        param = ConfParam(class_name, name, conf_type, description)
        # TODO check if param already exists
        if name not in REG_CONF_PARAMS_BY_NAME:
            REG_CONF_PARAMS_BY_NAME[name] = set()
        REG_CONF_PARAMS_BY_NAME[name].add(param)

        if class_name not in REG_CONF_PARAMS_BY_CLASS_NAME:
            REG_CONF_PARAMS_BY_CLASS_NAME[class_name] = set()
        REG_CONF_PARAMS_BY_CLASS_NAME[class_name].add(param)

        print(class_name, param.name)
        update_docstring(cls, param)

        return cls

    return _wrap_class


def update_docstring(cls, param):
    if cls.__doc__ is None:
        print(f'Warning: {cls.__name__} has no docstring')
        return
    cls.__doc__ += f'\n\nConfiguration parameter used:\n{param}'


def register_links(created_classes: Set[str]):
    """"""
    def _wrap_class(cls):
        class_name = cls.__name__
        # print(f'{class_name}: Registering link to {created_classes}\n')

        REG_CLASS_LINKS[class_name] = created_classes
        return cls

    return _wrap_class


class LinkParam():
    def __init__(self,
                 from_class: str,
                 choice_param: str,
                 choice_options: Dict[str, str],
                 description: str):
        self.from_class = from_class
        self.choice_param = choice_param
        self.choice_options = choice_options
        self.description = description

    def __str__(self):
        choices = '\n\t\t\t'.join(
                [f'{val} -> {dest}' for val, dest in self.choice_options.items()])
        return (f'\tName:   \t{self.choice_param}\n'
                f'\tType:   \tLink Parameter\n'
                f'\tChoices:\t{choices}\n'
                f'\tDescription:\t{self.description}')

def register_link_parameter(choice_param: str,
                            choice_options: Dict[str, str],
                            description: str = 'No description provided'):
    """"""
    def _wrap_class(cls):
        class_name = cls.__name__

        link_param = LinkParam(class_name, choice_param, choice_options, description)

        if class_name not in REG_LINK_PARAMS:
            REG_LINK_PARAMS[class_name] = set()
        REG_LINK_PARAMS[class_name].add(link_param)
        return cls

    return _wrap_class
    
