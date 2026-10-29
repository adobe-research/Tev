"""Executable v7.2 qualification gates."""

from importlib import import_module

_EXPORTS = {
    "qualify_corruption": (
        "training_data_collection.v72.qualification.corruption",
        "qualify_corruption",
    ),
    "qualify_world": ("training_data_collection.v72.qualification.world", "qualify_world"),
}


def __getattr__(name: str) -> object:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = ["qualify_corruption", "qualify_world"]
