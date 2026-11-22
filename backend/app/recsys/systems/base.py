"""System protocol and registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover
    from ..catalog import Catalog
    from ..query import UserQuery
    from ..result import Recommendation


@runtime_checkable
class System(Protocol):
    """A recommender. ``recommend`` sees only the :class:`UserQuery` and the catalogue."""

    name: str

    def config(self) -> dict[str, Any]:
        """Everything that determines the system's output (stored in results files)."""
        ...

    def recommend(self, query: UserQuery, k: int) -> Recommendation:
        """Top ``k`` routes for ``query``, best first, with a per-stage trace."""
        ...


Factory = Callable[..., System]


@dataclass(frozen=True)
class SystemSpec:
    name: str
    factory: Factory
    description: str = ""
    defaults: dict[str, Any] = field(default_factory=dict)
    uses_llm: bool = False  # factory accepts ``llm=`` (an app.llm client); the harness injects its own


_REGISTRY: dict[str, SystemSpec] = {}


def register_system(
    name: str, *, description: str = "", defaults: dict[str, Any] | None = None, uses_llm: bool = False
):
    """Decorator registering ``factory(catalog, **config)`` under ``name``."""

    def deco(factory: Factory) -> Factory:
        if name in _REGISTRY and _REGISTRY[name].factory is not factory:
            raise ValueError(f"system {name!r} already registered")
        _REGISTRY[name] = SystemSpec(name, factory, description, dict(defaults or {}), uses_llm)
        return factory

    return deco


def list_systems() -> list[str]:
    return sorted(_REGISTRY)


def get_system_spec(name: str) -> SystemSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown system {name!r}; registered: {', '.join(list_systems())}") from None


def build_system(name: str, catalog: Catalog, **config: Any) -> System:
    spec = get_system_spec(name)
    return spec.factory(catalog, **{**spec.defaults, **config})
