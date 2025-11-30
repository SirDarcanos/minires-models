# minires.py

import json
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import load_model
from xgboost import XGBRegressor 
from huggingface_hub import hf_hub_download


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

REPO_ID = "nicolamustone/minires"

NN_FILENAME   = "minires.keras"
XGB_FILENAME  = "minires_xgb.json"
META_FILENAME = "minires_meta.json"


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class minires:
    """
    minires: ensemble predictor for resin usage (grams).

    Downloads artifacts from Hugging Face Hub (repo: nicolamustone/minires)
    and caches them locally via huggingface_hub.
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        revision: str | None = None,
        verbose: int = 0,
    ):
        download_kwargs = {}
        if cache_dir is not None:
            download_kwargs["local_dir"] = cache_dir
            download_kwargs["local_dir_use_symlinks"] = False
        if revision is not None:
            download_kwargs["revision"] = revision

        self.nn_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=NN_FILENAME,
            **download_kwargs,
        )
        self.xgb_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=XGB_FILENAME,
            **download_kwargs,
        )
        self.meta_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=META_FILENAME,
            **download_kwargs,
        )

        self.nn = load_model(self.nn_path, compile=False)
        self.xgb = XGBRegressor()
        
        self.xgb.load_model(self.xgb_path)

        with open(self.meta_path, "r") as f:
            meta = json.load(f)

        self.w_nn: float = float(meta["w_nn"])
        self.features: list[str] = list(meta["features"])
        self.verbose: int = verbose

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
        """
        Xp = self._prepare_X(X)
        v = self.verbose if verbose is None else verbose

        nn_pred = self.nn.predict(Xp, verbose=v).flatten()
        xgb_pred = self.xgb.predict(Xp)

        return self.w_nn * nn_pred + (1.0 - self.w_nn) * xgb_pred
