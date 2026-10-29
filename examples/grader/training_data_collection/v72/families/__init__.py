"""V7.2 family implementation boundary."""

__all__ = ["FamilyImplementation", "GovernedSourceView", "WorldStateView"]


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(name)
    from .base import FamilyImplementation, GovernedSourceView, WorldStateView

    return {
        "FamilyImplementation": FamilyImplementation,
        "GovernedSourceView": GovernedSourceView,
        "WorldStateView": WorldStateView,
    }[name]
