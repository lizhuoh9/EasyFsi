from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.validation.ansys_vertical_flap import validation_artifact_hygiene as _impl  # noqa: E402
from tools.validation.ansys_vertical_flap.validation_artifact_hygiene import *  # noqa: F401,F403,E402


def __getattr__(name: str):
    return getattr(_impl, name)


if __name__ == "__main__":
    raise SystemExit(main())
