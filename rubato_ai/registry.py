"""
Keep a global registry of hyperparameters used.

Contain functions that are used as decorators for classes to register
their usage of configuration parameters. This allows us to declare configuration parameters,
in the code not far from the actual usage of the parameter,
and see which classes use which parameters.

In addition, we define the type that the setting of a config parameter should have,
as well as a description (see :meth:`register_param`).

Not only can we register which classes access which config parameter,
we can also register that classes initialize other classes that access config parameters.
These links are registered using :meth:`register_links`.

Some classes may define links that in which one of multiple target classes
is linked to, depending on the value of a configuration parameter.
These are called link parameters, and are registered with :meth:`register_link_param`

Since the decorators run at import time, we populate the register as
soon as the relevant classes are imported.
This allows to change the docstrings of those classes at import time,
documenting which parameters are used just by registering them.
This may save some effort when writing that pesky documentation.

.. seealso::
    Module :py:mod:`.config_check`
        Validate a configuration dict based on the registry
"""
import os
import sys

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
        used_by = ', '.join(f':class:`{p.class_name}`' for p in REG_CONF_PARAMS_BY_NAME[self.name])
        return (f'    =========== ==================\n'
                f'    Name        ``{self.name}``\n'
                f'    Used by     {used_by}\n'
                f'    Type        ``{self.conf_type}``\n'
                f'    Description {self.description}\n'
                f'    =========== ==================\n')


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
        choices = '\n\n                '.join(f'``\'{k}\'`` -> :class:`{v}`' for k, v in self.choice_options.items())
        return (f'    =========== ==================\n'
                f'    Name        ``{self.choice_param}``\n'
                f'    Type        Link Parameter\n'
                f'    Choices     {choices}\n'
                f'    Description {self.description}\n'
                f'    =========== ==================\n')


def register_param(name: str,
                   conf_type: Type,
                   description: str = 'No description provided'):
    """
    Register a hyperparameter to the parameter registry.
    Use as class decorator to define the config parameters used by that class.

    Example:
        Here is how we would register the use of the config parameter ``'sequence_length'``
        for a class::

            @register_param('sequence_length', int,
                            'Maximum input sequence length')
            class MultiHeadAttention(...):
                ...

    After adding the parameter to the registry, an overview of the parameter
    is appended to the class docstring, formatted in reStructuredText.

    .. todo::
        handle two classes using same parameter (check the type or sth)
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

        _add_param_docstring(cls, param)

        return cls

    return _wrap_class


def register_link_param(choice_param: str,
                        choice_options: Dict[str, str],
                        description: str = 'No description provided'):
    """
    A link parameter links from a source class to a target class in which
    the target class depends on the value of a config parameter.

    Example:
        The class :class:`TransformerBlock` defines a link parameter with the name ``'attn_type'``,
        allowing the type of attention layer to be selected. When ``config['attn_type']``
        is set to ``'relative'``, the :class:`TransformerBlock` creates a :class:`RelativeGlobalAttention`
        layer. When set to ``'absolute'``, a :class:`MultiHeadAttention` layer is created instead.

        This relation is defined using this function as decorator::

            @register_link_parameter('attn_type', {
                'absolute':'MultiHeadAttention',
                'relative':'RelativeGlobalAttention'
                }
            )
            class TransformerBlock(...):
                ...

    After adding the link parameter to the registry, an overview of the link parameter
    is appended to the class docstring, formatted in reStructuredText.

    For unconditional links, see :meth:`register_links`
    """
    def _wrap_class(cls):
        class_name = cls.__name__

        link_param = LinkParam(class_name, choice_param, choice_options, description)

        if class_name not in REG_LINK_PARAMS:
            REG_LINK_PARAMS[class_name] = set()
        REG_LINK_PARAMS[class_name].add(link_param)

        _add_param_docstring(cls, link_param)

        return cls

    return _wrap_class
    
def register_links(created_classes: Set[str]):
    """
    When a link is registered to another class name, this indicates that all
    parameters of the target class should be checked whenever the parameters
    of the source class are checked.

    The two cases for this are when a class initializes another class
    (such as a ``keras.Layer`` creating sublayers), and a class inheriting from a superclass
    (where the superclass accesses its own config parameters or defines its own config links).

    You should use the decorator ``@register_links`` at most once per class.

    Example:
        Because :class:`RelativeGlobalAttention` inherits from :class:`MultiHeadAttention`,
        a link is registered from ``'RelativeGlobalAttention'`` to ``'MultiHeadAttention'``,
        since configuration parameters of MHA are always used whenever RGA is used::

            @register_links({'MultiHeadAttention'})
            class RelativeGlobalAttention(MultiHeadAttention):
                ...

    After adding the links to the registry, a description of the link is appended to the class docstring.
    This link description is formatted with reStructuredText.

    For optional links, see :meth:`register_link_param`.
    """
    def _wrap_class(cls):
        class_name = cls.__name__
        # print(f'{class_name}: Registering link to {created_classes}\n')

        REG_CLASS_LINKS[class_name] = created_classes

        _add_link_docstring(cls, created_classes)
        return cls

    return _wrap_class


def _add_param_docstring(cls, param: Union[ConfParam, LinkParam]):
    """
    Add registered parameter information to the class docstring

    Since the registry is populated at import time, this can be used
    to change auto-generated documentation like sphinx
    """
    if cls.__doc__ is None:
        print(f'Warning: class {cls.__name__} has no docstring', file=sys.stderr)
        return
    cls.__doc__ += f'\n\n    | Configuration parameter used:\n\n{param}\n'

def _add_link_docstring(cls, targets: Set[str]):
    """
    Add linked classses to docstring. Links are formatted for sphinx
    """
    if cls.__doc__ is None:
        print(f'Warning: class {cls.__name__} has no docstring', file=sys.stderr)
        return
    link_fmt = ', '.join([f':class:`{link}`' for link in targets])
    cls.__doc__ += f'\n\n    | Class links to: {link_fmt}\n\n'
