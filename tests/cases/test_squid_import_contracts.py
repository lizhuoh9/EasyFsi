from __future__ import annotations

import subprocess
import sys
import unittest


class SquidImportContractTests(unittest.TestCase):
    def test_spec_and_coupling_modules_import_in_a_clean_interpreter(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import cases.squid_soft_robot.spec; "
                    "import cases.squid_soft_robot.coupling_common"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
