#!/usr/bin/env python3
# Copyright (c) Facebook. and its affiliates. All Rights Reserved

import six, inspect

class Registry(object):
    """
    The registry that provides name -> object mapping, to support third-party
      users' custom modules.

    To create a registry (inside detectron2):
        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:
        @BACKBONE_REGISTRY.register("MyBackbone")
        class MyBackbone():
            ...
    Or:
        BACKBONE_REGISTRY.register(name="MyBackbone", obj=MyBackbone)
    """

    def __init__(self, name, allow_override=False):
        """
        Args:
            name (str): the name of this registry
        """
        self._name = name
        self._allow_override = allow_override

        self._obj_map = {}

    def _do_register(self, name, obj):
        if name in self._obj_map and not self._allow_override:
            raise ValueError(
                "An object named '{}' was already registered in '{}' registry!".format(
                    name, self._name
                )
            )
        self._obj_map[name] = obj

    def register(self, name=None, obj=None):
        """
        Register the given object under the the name or `obj.__name__` if name is None.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class):
                nonlocal name
                if name is None:
                    name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        if name is None:
            name = obj.__name__
        self._do_register(name, obj)

    def register_dict(self, mapping):
        """
        Register a dict of objects
        """
        assert isinstance(mapping, dict)
        [self.register(name, obj) for name, obj in mapping.items()]

    def get(self, name, is_raise=True):
        """
        Raise an exception if the key is not found if `is_raise` is True,
          return None otherwise
        """
        ret = self._obj_map.get(name)
        if ret is None and is_raise:
            raise KeyError(
                "No object named '{}' found in '{}' registry! Available names: {}".format(
                    name, self._name, list(self._obj_map.keys())
                )
            )
        return ret

    def get_names(self):
        return self._obj_map.keys()

    def items(self):
        return self._obj_map.items()

    def __len__(self):
        return len(self._obj_map)

    def keys(self):
        return self._obj_map.keys()

    def __contains__(self, key):
        return key in self._obj_map

    def __getitem__(self, key):
        return self._obj_map[key]


def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        obj: The constructed object.
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()
    obj_type = args.pop('type')
    if isinstance(obj_type, six.string_types):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError('{} is not in the {} registry'.format(
                obj_type, registry.name))
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_cls(**args)

