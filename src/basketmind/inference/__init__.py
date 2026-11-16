"""BasketMind inference.

The data-only and precomputed services import no torch. `BasketMindService` does, so it loads on first use.
"""

from basketmind.inference.data_service import DataService, NotFound, PrecomputedService, Recommendation

__all__ = ["BasketMindService", "DataService", "NotFound", "PrecomputedService", "Recommendation"]


def __getattr__(name: str):
    if name == "BasketMindService":
        from basketmind.inference.service import BasketMindService

        return BasketMindService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
