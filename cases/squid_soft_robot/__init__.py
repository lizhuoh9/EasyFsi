from __future__ import annotations

from . import checkpointing as _checkpointing
from . import cli as _cli
from . import diagnostics as _diagnostics
from . import history as _history
from . import outputs as _outputs
from . import runner as _runner
from . import schedules as _schedules
from . import snapshots as _snapshots
from . import source_config as _source_config
from . import spec as _spec


_EXPORT_MODULES = (
    _runner,
    _cli,
    _spec,
    _source_config,
    _schedules,
    _history,
    _checkpointing,
    _diagnostics,
    _snapshots,
    _outputs,
)


for _module in _EXPORT_MODULES:
    for _name in dir(_module):
        if not (_name.startswith("__") and _name.endswith("__")):
            globals()[_name] = getattr(_module, _name)


def __getattr__(name: str):
    for _module in _EXPORT_MODULES:
        if hasattr(_module, name):
            return getattr(_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    names = set(globals())
    for _module in _EXPORT_MODULES:
        names.update(dir(_module))
    return sorted(names)


del _module
del _name
