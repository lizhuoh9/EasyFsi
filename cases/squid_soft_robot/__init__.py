from __future__ import annotations

# Transitional compatibility layer while the historical single-file squid case
# is being split. New code should import from explicit submodules.

from . import checkpointing as _checkpointing
from . import cli as _cli
from . import coupling_common as _coupling_common
from . import coupling_legacy as _coupling_legacy
from . import coupling_sharp as _coupling_sharp
from . import diagnostics as _diagnostics
from . import fluid_step as _fluid_step
from . import history as _history
from . import outputs as _outputs
from . import rows as _rows
from . import runner as _runner
from . import runtime_state as _runtime_state
from . import schedules as _schedules
from . import setup as _setup
from . import snapshots as _snapshots
from . import solid_step as _solid_step
from . import source_config as _source_config
from . import spec as _spec
from . import step_context as _step_context
from . import step_loop as _step_loop
from . import summary as _summary
from . import trial_replay as _trial_replay


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
    _runtime_state,
    _setup,
    _summary,
    _rows,
    _step_context,
    _step_loop,
    _trial_replay,
    _solid_step,
    _fluid_step,
    _coupling_common,
    _coupling_legacy,
    _coupling_sharp,
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
