from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.validation.ansys_vertical_flap import fluent_parity as _impl  # noqa: E402
from tools.validation.ansys_vertical_flap.fluent_parity import *  # noqa: F401,F403,E402


_candidate_blockers = _impl._candidate_blockers
_candidate_status = _impl._candidate_status
_parity_metrics = _impl._parity_metrics
_relative_comparison = _impl._relative_comparison
_sha256_file = _impl._sha256_file
_validate_active_contract_manifest = _impl._validate_active_contract_manifest


def __getattr__(name: str):
    return getattr(_impl, name)


def main() -> int:
    try:
        payload = _impl.run()
    except Exception as exc:  # pragma: no cover - command-line failure path
        print(
            f"[traction_selected_formulation_fluent_parity] ERROR: {exc}",
            file=sys.stderr,
        )
        return 1
    print(
        "[traction_selected_formulation_fluent_parity] wrote "
        f"{payload['candidate_status']} to {_impl.OUTPUT_DIR}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
