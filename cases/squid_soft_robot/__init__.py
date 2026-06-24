from __future__ import annotations

import sys as _sys

from . import runner as _runner


_sys.modules[__name__] = _runner
