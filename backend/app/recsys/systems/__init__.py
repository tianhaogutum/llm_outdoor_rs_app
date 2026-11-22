"""Systems registry: ``name -> factory(catalog, **config) -> System``.

Importing this package registers the built-in systems (``random``, ``v0_rules``). Phase 3 adds its
stages as new modules and registers them the same way::

    @register_system("v1_filters", description="v0_rules behind the hard filter")
    def _make(catalog, **cfg):
        return MySystem(catalog, **cfg)
"""

from . import (
    baselines,  # noqa: E402,F401  (registers random and v0_rules)
    pipelines,  # noqa: E402,F401  (registers the Phase 3 pipelines)
)
from .base import System, SystemSpec, build_system, get_system_spec, list_systems, register_system

__all__ = ["System", "SystemSpec", "build_system", "get_system_spec", "list_systems", "register_system"]
