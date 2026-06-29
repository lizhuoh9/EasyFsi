from __future__ import annotations

import unittest
from pathlib import Path


GATE_DOC = (
    Path("docs")
    / "validation"
    / "ANSYS_VERTICAL_FLAP_REAL_FLUENT_IMPORT_GATE_2026-06-29.md"
)


class AnsysVerticalFlapRealFluentImportGateDocTests(unittest.TestCase):
    def test_real_fluent_import_gate_blocks_premature_parity_claims(self):
        text = GATE_DOC.read_text(encoding="utf-8")

        for phrase in (
            "four CSV source exports",
            "`step = 50`",
            "source document",
            "run id",
            "author",
            "date",
            "must not contain `EasyFsi`",
            "must not contain `HIBM-MPM`",
            "collection validator",
            "complete real reference coverage",
            "active manifest promotion",
            "cannot claim Fluent parity",
        ):
            self.assertIn(phrase, text)


if __name__ == "__main__":
    unittest.main()
