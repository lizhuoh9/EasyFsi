from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.validation.ansys_vertical_flap import fluent_artifact_policy as _impl  # noqa: E402
from tools.validation.ansys_vertical_flap.fluent_artifact_policy import *  # noqa: F401,F403,E402


def __getattr__(name: str):
    return getattr(_impl, name)


if __name__ == "__main__":
    raise SystemExit(main())
