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
This can be done with the decorator :meth:`document_registrations`.

.. seealso::
    Module :py:mod:`.config_check`
        Validate a configuration dict based on the registry

.. todo::
    - Document breaks_compatibility
    - Move typevars to their own class?
"""
import os

from typing import Dict, Set, Union, Type, Any

PathLike = Union[str, bytes, os.PathLike]
ConfDict = Dict[str, Any]


class ConfigRegistry:
    """
    .. todo::
        document this
    """
    def __init__(self):
        self.conf_params_by_name: Dict[str, Set['ConfParam']] = {}
        """All config parameters, accessed by parameter name"""

        self.conf_params_by_class_name: Dict[str, Set['ConfParam']] = {}
        """All config parameters, accessed by class name"""

        self.class_links: Dict[str, Set[str]] = {}
        """Set of classes created by key class during initialization,
        including possibly a superclass"""

        self.link_params: Dict[str, Set['LinkParam']] = {}
        """Link parameters accesses by class name"""

        self.ckpt_incompatible_params: Set[str] = set()
        """Config parameters that break checkpoint compatibility with a change in value"""

    def _register_param(self, class_name: str, name: str, conf_type: Type,
                        description: str, breaks_compatibility: bool):
        param = ConfParam(class_name, name, conf_type, description, breaks_compatibility)
        # TODO check if param already exists
        if name not in self.conf_params_by_name:
            self.conf_params_by_name[name] = set()
        self.conf_params_by_name[name].add(param)

        if class_name not in self.conf_params_by_class_name:
            self.conf_params_by_class_name[class_name] = set()
        self.conf_params_by_class_name[class_name].add(param)

        if breaks_compatibility:
            self.ckpt_incompatible_params.add(name)

    def _register_link_param(self, class_name: str, choice_param: str,
                             choice_options: Dict[str, str],
                             description: str,
                             breaks_compatibility: bool):
        link_param = LinkParam(class_name, choice_param, choice_options, description,
                               breaks_compatibility)

        if class_name not in self.link_params:
            self.link_params[class_name] = set()
        self.link_params[class_name].add(link_param)

        if breaks_compatibility:
            self.ckpt_incompatible_params.add(choice_param)

    def _register_links(self, class_name: str, created_classes: Set[str]):
        self.class_links[class_name] = created_classes

    def generate_config_docstring(self, class_name: str) -> str:
        """
        Create a (doc)string containing info about all config parameters,
        link parameters and links registered by a class.
        """
        docstring = ''
        # ConfParam info
        if class_name in self.conf_params_by_class_name:
            docstring += '\n\n\n    **Config parameters used**:\n'
            for param in self.conf_params_by_class_name[class_name]:
                docstring += f'\n{param}\n'
        # LinkParam info
        if class_name in self.link_params:
            docstring += '\n\n\n    **Link parameters**:\n'
            for param in self.link_params[class_name]:
                docstring += f'\n{param}\n'
        # Linked classes
        if class_name in self.class_links:
            link_fmt = ', '.join([f':class:`{link}`' for link in self.class_links[class_name]])
            docstring += f'\n\n    | **Class links to**: {link_fmt}\n\n'

        return docstring
    
    def breaks_checkpoint_compatibility(self, parameter_name: str) -> bool:
        """
        Return whether a parameter has been registered anywhere with the setting
        ``breaks_compatibility = True``.
        """
        return parameter_name in self.ckpt_incompatible_params


CONFIG_REGISTRY = ConfigRegistry()
"""'Global' instance of the config registry that can be populated with config parameters"""


class ConfParam:
    def __init__(self,
                 class_name: str,
                 name: str,
                 conf_type: Type,
                 description: str,
                 breaks_compatibility: bool):
        self.class_name = class_name
        self.name = name
        self.conf_type = conf_type
        self.description = description
        self.breaks_compatibility = breaks_compatibility

    def __str__(self):
        # TODO pass CONFIG_REGISTRY as parameter?
        used_by = ', '.join(f':class:`{p.class_name}`' for p in CONFIG_REGISTRY.conf_params_by_name[self.name])
        compat = ('**Breaks**' if self.breaks_compatibility else 'Doesn\'t break')
        return (f'    =========== ==================\n'
                f'    Name        ``{self.name}``\n'
                f'    Used by     {used_by}\n'
                f'    Type        ``{self.conf_type}``\n'
                f'    Description {self.description}\n'
                f'    Compatible  {compat} checkpoint compatibility\n'
                f'    =========== ==================\n')


class LinkParam():
    def __init__(self,
                 from_class: str,
                 choice_param: str,
                 choice_options: Dict[str, str],
                 description: str,
                 breaks_compatibility: bool):
        self.from_class = from_class
        self.choice_param = choice_param
        self.choice_options = choice_options
        self.description = description
        self.breaks_compatibility = breaks_compatibility

    def __str__(self):
        choices = '\n\n                ' \
                .join(f'``\'{k}\'`` -> :class:`{v}`' for k, v in self.choice_options.items())
        compat = ('Breaks' if self.breaks_compatibility else 'Doesn\'t break')
        return (f'    =========== ==================\n'
                f'    Name        ``{self.choice_param}``\n'
                f'    Type        Link Parameter\n'
                f'    Choices     {choices}\n'
                f'    Description {self.description}\n'
                f'    Compatible  {compat} checkpoint compatibility\n'
                f'    =========== ==================\n')


def register_param(name: str,
                   conf_type: Type,
                   description: str = 'No description provided',
                   breaks_compatibility: bool = False):
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

    .. todo::
        handle two classes using same parameter (check the type or sth)
    """

    def _wrap_class(cls):
        class_name = cls.__name__
        # print(f'{class_name}: Registering config parameter \'{name}\'')
        # print(f'\tType: {conf_type}\n\tDescription: {description}\n\tDefault: {default}\n')
        CONFIG_REGISTRY._register_param(class_name, name, conf_type, description,
                                        breaks_compatibility)
        return cls

    return _wrap_class


def register_link_param(choice_param: str,
                        choice_options: Dict[str, str],
                        description: str = 'No description provided',
                        breaks_compatibility: bool = True):
    """
    A link parameter links from a source class to a target class in which
    the target class depends on the value of a config parameter.

    Example:
        The class :class:`TransformerBlock` defines a link parameter with the name ``'attn_type'``,
        allowing the type of attention layer to be selected. When ``config['attn_type']``
        is set to ``'relative'``, the :class:`TransformerBlock` creates a :class:`RelativeGlobalAttention`
        layer. When set to ``'absolute'``, a :class:`MultiHeadAttention` layer is created instead.

        This relation is defined using this function as decorator::

            @register_link_param('attn_type', {
                'absolute':'MultiHeadAttention',
                'relative':'RelativeGlobalAttention'
                }
            )
            class TransformerBlock(...):
                ...

    For unconditional links, see :meth:`register_links`
    """
    def _wrap_class(cls):
        class_name = cls.__name__
        CONFIG_REGISTRY._register_link_param(class_name, choice_param, choice_options, 
                                             description, breaks_compatibility)
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

    For optional links, see :meth:`register_link_param`.
    """
    def _wrap_class(cls):
        class_name = cls.__name__
        # print(f'{class_name}: Registering link to {created_classes}\n')
        CONFIG_REGISTRY._register_links(class_name, created_classes)
        return cls

    return _wrap_class


def document_registrations(cls):
    """
    Decorator to add information about registered parameters and links
    to a class docstring.

    Since the registry is populated at import time, this can be used
    to populate auto-generated documentation with config parameters used.
    
    This decarator takes no arguments and should be put before the other registry decorators:

    .. code-block:: python
       :linenos:
       :emphasize-lines: 1
       
       @document_registrations
       @register_param(...)
       @register_link_param(...)
       @register_links(...)
       class MyClass(...):
           ...
    """
    # Decorator without arguments doesn't need the wrapper function
    class_name = cls.__name__
    if cls.__doc__ is None:
        cls.__doc__ = ''

    cls.__doc__ += CONFIG_REGISTRY.generate_config_docstring(class_name)
    return cls
