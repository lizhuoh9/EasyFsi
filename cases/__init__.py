from __future__ import annotations

CASE_MODULES = {
    "squid-soft-robot": "cases.squid_soft_robot",
}
AVAILABLE_CASES = tuple(CASE_MODULES)

__all__ = ["AVAILABLE_CASES", "CASE_MODULES"]
