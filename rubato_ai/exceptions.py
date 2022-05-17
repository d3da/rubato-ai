from registry import ConfParam, LinkParam

from typing import Any

class ConfigException(Exception):
    pass


class ConfParamException(ConfigException):
    def __init__(self, msg: str, param: ConfParam):
        super().__init__(f'{msg}\n{param}\n')

class ConfParamUnsetError(ConfParamException):
    def __init__(self, param: ConfParam):
        super().__init__('Configuration parameter unset.', param)

class ConfParamTypeError(ConfParamException):
    def __init__(self, param: ConfParam, value: Any):
        super().__init__(f'Configuration parameter has wrong type. (got {type(value)})', param)


class LinkParamException(ConfigException):
    def __init__(self, msg: str, link_param: LinkParam):
        super().__init__(f'{msg}\n{link_param}\n')

class LinkParamUnsetError(LinkParamException):
    def __init__(self, link_param: LinkParam):
        super().__init__('Configuration parameter unset.', link_param)

class LinkParamWrongValueError(LinkParamException):
    def __init__(self, link_param: LinkParam, value: Any):
        super().__init__(f'Configuration parameter has wrong value. (got {value})', link_param)


class CheckpointIncompatibleError(ConfigException):
    def __init__(self, param_name: str, saved_value: Any, curr_value: Any):
        super().__init__(f'Changing parameter \'{param_name}\' from {saved_value} to {curr_value} '
                         'has broken checkpoint compatibility')
