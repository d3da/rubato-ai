#!/usr/bin/env python3
"""
Keep a global registry of hyperparameters used.

Register a parameter directly accessed in a class by decorating the class with @register_param.
If a class creates other classes (having their own parameters) in its __init__ method, decorate the class with @register_creates.
"""
from typing import Dict, Set

"""
TODO register params in: optimizer.py, input_loader, midi_processor, performance_rnn.py
TODO make the register a class

TODO register class name, description, default of choice parameters

TODO handle superclasses separately from creates classes
TODD make start class that creates PerformanceRNNModel or TransformerModel
"""

class ConfParam:
    def __init__(self,
                 class_name: str,
                 name: str,
                 conf_type: str,
                 default: object,
                 description: str):
        self.class_name = class_name
        self.name = name
        self.conf_type = conf_type
        self.default = default
        self.description = description


# All config parameters, accessed by parameter name
CONFIG_REG_BY_NAME: Dict[str, Set[ConfParam]] = {}

# All config parameters, accessed by class name
CONFIG_REG_BY_CLASS_NAME: Dict[str, Set[ConfParam]] = {}

# Set of classes created by key class during initialization, including possibly a superclass
CONFIG_REG_CLASS_CREATES: Dict[str, Set[str]] = {}

# Maps class_name -> { optional parameter name : { option value : linked classname }}
CONFIG_REG_OPTIONAL_CREATES: Dict[str, Dict[str, Dict[str, str]]] = {}


def register_param(name: str,
                   conf_type: str,
                   default: object = None,
                   description: str = 'No description provided'):
    """
    Register a hyperparameter to the parameter registry.
    Use as class decorator to define the config parameters used by that class.

    TODO register (optional) links between class parameters
    TODO handle two classes using same parameter
    TODO change config value to default if not present (warn the user)
    """

    def _wrap_class(cls):
        class_name = cls.__name__
        # print(f'{class_name}: Registering config parameter \'{name}\'')
        # print(f'\tType: {conf_type}\n\tDescription: {description}\n\tDefault: {default}\n')

        param = ConfParam(class_name, name, conf_type, default, description)
        # TODO check if param already exists
        if name not in CONFIG_REG_BY_NAME:
            CONFIG_REG_BY_NAME[name] = set()
        CONFIG_REG_BY_NAME[name].add(param)

        if class_name not in CONFIG_REG_BY_CLASS_NAME:
            CONFIG_REG_BY_CLASS_NAME[class_name] = set()
        CONFIG_REG_BY_CLASS_NAME[class_name].add(param)

        return cls

    return _wrap_class


def register_creates(created_classes: Set[str]):
    """"""
    def _wrap_class(cls):
        class_name = cls.__name__
        print(f'{class_name}: Registering link to {created_classes}\n')

        CONFIG_REG_CLASS_CREATES[class_name] = created_classes
        return cls

    return _wrap_class

def register_optional_creates(choice_param: str,
                              choice_options: Dict[str, str]):
    """"""
    def _wrap_class(cls):
        class_name = cls.__name__
        print(f'{class_name}: Registering optional link under parameter \'{choice_param}\'')
        for name, class_link in choice_options.items():
            print(f'\t\'{name}\' -> {class_link}')
        print()

        if class_name not in CONFIG_REG_OPTIONAL_CREATES:
            CONFIG_REG_OPTIONAL_CREATES[class_name] = {}
        CONFIG_REG_OPTIONAL_CREATES[class_name][choice_param] = choice_options
        return cls

    return _wrap_class
    
