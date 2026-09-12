from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile


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
                        "from minires.preparation.slicing_contract import BUNDLED_PROFILE_RESOURCE; "
                        "from hashlib import sha256; "
                        "print(json.dumps({'package': minires.__name__, "
                        "'profile': sha256(BUNDLED_PROFILE_RESOURCE.read_bytes()).hexdigest()}))"
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
        from minires.preparation.slicing_contract import BUNDLED_PROFILE_RESOURCE, PROFILE_SHA256

        self.assertEqual(sha256(BUNDLED_PROFILE_RESOURCE.read_bytes()).hexdigest(), PROFILE_SHA256)
        self.assertTrue(BUNDLED_PROFILE_RESOURCE.is_file())

    def test_profile_can_be_materialized_from_a_non_filesystem_loader(self):
        package_root = Path(__file__).parents[1] / "src" / "minires"
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "minires.zip"
            with ZipFile(archive, "w") as package:
                for source in package_root.rglob("*"):
                    if source.is_file():
                        package.write(source, Path("minires") / source.relative_to(package_root))
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(archive)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from hashlib import sha256\n"
                        "from importlib.resources import as_file\n"
                        "from minires.preparation.slicing_contract import BUNDLED_PROFILE_RESOURCE\n"
                        "with as_file(BUNDLED_PROFILE_RESOURCE) as path:\n"
                        "    print(sha256(path.read_bytes()).hexdigest())\n"
                    ),
                ],
                cwd=directory,
                env=environment,
                text=True,
                capture_output=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            completed.stdout.strip(),
            "06acac3fe2a3d762fb56ec2d1bde58fe9e15104556091438c81c4e90131d2d0e",
        )


if __name__ == "__main__":
    unittest.main()
