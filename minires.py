# minires.py

import json
import joblib
import numpy as np
import pandas as pd
import urllib.request
from pathlib import Path
from tensorflow.keras.models import load_model


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "minires"

NN_FILENAME   = "minires.keras"
XGB_FILENAME  = "minires.joblib"
META_FILENAME = "minires.json"

REMOTE_FILES = {
    "nn":   f"https://buthonestly.io/minires_w/{NN_FILENAME}",
    "xgb":  f"https://buthonestly.io/minires_w/{XGB_FILENAME}",
    "meta": f"https://buthonestly.io/minires_w/{META_FILENAME}",
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
        f.write(r.read())


def _ensure_cached(filename: str, url: str, cache_dir: Path) -> Path:
    path = cache_dir / filename
    if not path.exists():
        if url is None:
            raise FileNotFoundError(f"{path} not found and no URL provided")
        _download_file(url, path)
    return path


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class minires:
    """
    minires: ensemble predictor for resin usage (grams).

    Internally it loads:
      - a Keras NN (.keras)
      - an XGBoost model (.joblib)
      - a JSON meta file with:
          { "w_nn": float, "features": [feature names...] }

    On first use it downloads the artifacts into a cache directory
    (~/.cache/minires by default). Subsequent runs load from cache.
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        nn_url: str | None = None,
        xgb_url: str | None = None,
        meta_url: str | None = None,
        verbose: int = 0,
    ):
        cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR

        nn_url   = nn_url   or REMOTE_FILES["nn"]
        xgb_url  = xgb_url  or REMOTE_FILES["xgb"]
        meta_url = meta_url or REMOTE_FILES["meta"]

        self.nn_path = _ensure_cached(NN_FILENAME,   nn_url,   cache_dir)
        self.xgb_path = _ensure_cached(XGB_FILENAME, xgb_url,  cache_dir)
        self.meta_path = _ensure_cached(META_FILENAME, meta_url, cache_dir)

        self.nn = load_model(self.nn_path)
        self.xgb = joblib.load(self.xgb_path)

        with open(self.meta_path, "r") as f:
            meta = json.load(f)

        self.w_nn: float = float(meta["w_nn"])
        self.features: list[str] = list(meta["features"])
        self.verbose: int = verbose  # default verbosity for NN predict

        # Print which ensemble weights are used
        if self.verbose:
            w_xgb = 1.0 - self.w_nn
            print(f"[minires] ensemble weights: NN={self.w_nn:.3f}, XGB={w_xgb:.3f}")

    # ---------------------- internal helpers ---------------------- #

    def _prepare_X(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X
        else:
            df = pd.DataFrame(X, columns=self.features)

        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        return df[self.features]

    # -------------------------- public API ------------------------ #

    def predict(self, X, verbose: int | None = None) -> np.ndarray:
        """
        Predict resin usage (grams) for the given samples.

        Parameters
        ----------
        X : pandas.DataFrame or array-like
            Must contain the feature columns listed in self.features
            (or be in that exact order if array-like).
        verbose : int or None
            Verbosity argument passed to Keras model.predict().
            If None, uses the default set at __init__.

        Returns
        -------
        np.ndarray
            1D array of predicted resin grams.
        """
        Xp = self._prepare_X(X)

        v = self.verbose if verbose is None else verbose

        nn_pred = self.nn.predict(Xp, verbose=v).flatten()
        xgb_pred = self.xgb.predict(Xp)

        return self.w_nn * nn_pred + (1.0 - self.w_nn) * xgb_pred
