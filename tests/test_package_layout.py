from hashlib import sha256
import json
import os
import subprocess
import sys
import tempfile
import unittest


class InstalledPackageLayoutTests(unittest.TestCase):
    def test_installed_package_and_profile_load_outside_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = os.environ.copy()
            environment.pop("PYTHONPATH", None)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import json, minires; "
                        "from minires.preparation.slicing_contract import BUNDLED_PROFILE_PATH; "
                        "from hashlib import sha256; "
                        "print(json.dumps({'package': minires.__name__, "
                        "'profile': sha256(BUNDLED_PROFILE_PATH.read_bytes()).hexdigest()}))"
                    ),
                ],
                cwd=directory,
                env=environment,
                text=True,
                capture_output=True,
                check=True,
            )

        result = json.loads(completed.stdout)
        self.assertEqual(result["package"], "minires")
        self.assertEqual(
            result["profile"],
            "06acac3fe2a3d762fb56ec2d1bde58fe9e15104556091438c81c4e90131d2d0e",
        )

    def test_profile_bytes_match_the_pinned_source_resource(self):
        from minires.preparation.slicing_contract import BUNDLED_PROFILE_PATH, PROFILE_SHA256

        self.assertEqual(sha256(BUNDLED_PROFILE_PATH.read_bytes()).hexdigest(), PROFILE_SHA256)
        self.assertTrue(BUNDLED_PROFILE_PATH.is_file())


if __name__ == "__main__":
    unittest.main()
