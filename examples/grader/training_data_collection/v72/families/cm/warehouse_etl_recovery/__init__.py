"""Benchmark-shaped warehouse ETL family for v7.2."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .definition import WAREHOUSE_ETL_RECOVERY_FAMILY, WarehouseEtlRecoveryFamily

__all__ = ["WAREHOUSE_ETL_RECOVERY_FAMILY", "WarehouseEtlRecoveryFamily"]


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(name)
    from .definition import WAREHOUSE_ETL_RECOVERY_FAMILY, WarehouseEtlRecoveryFamily

    return {
        "WAREHOUSE_ETL_RECOVERY_FAMILY": WAREHOUSE_ETL_RECOVERY_FAMILY,
        "WarehouseEtlRecoveryFamily": WarehouseEtlRecoveryFamily,
    }[name]
