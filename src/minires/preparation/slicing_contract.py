"""Shared immutable provenance for MiniRes sliced-resin-mass labels."""

from importlib.resources import files
from pathlib import Path

EBMINIMANAGER_REVISION = "1a841195813136ee3b380ab1d192727f385f7a55"
PROFILE_RELATIVE_PATH = Path("prediction/config-anycubic-mono.ini")
BUNDLED_PROFILE_PATH = Path(
    str(files("minires.preparation.profiles").joinpath("config-anycubic-mono.ini"))
)
PROFILE_SHA256 = "06acac3fe2a3d762fb56ec2d1bde58fe9e15104556091438c81c4e90131d2d0e"
DENSITY_G_PER_ML = 1.1
LAYER_HEIGHT_MM = 0.05
SLICER_ADDED_SUPPORTS = False
