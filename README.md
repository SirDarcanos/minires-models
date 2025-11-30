# MiniRes: Resin Usage Predictor for 3D Printed Miniatures

MiniRes is a small Python library + pretrained ensemble model that predicts **resin usage in grams** for pre-supported 3D printed miniatures.

It takes tabular features exported from your slicing & analysis pipeline (UVtools, PrusaSlicer, trimesh, etc.) and returns a single float per part: the estimated resin usage in grams.

Under the hood, MiniRes is an ensemble model with:

- A **Keras neural network** +
- An **XGBoost regressor**

The model weights are hosted on Hugging Face and downloaded/cached automatically.

Hugging Face model repo:  
https://huggingface.co/nicolamustone/minires

The repositoriy above includes a helper for creating a suitable CSV of data from STL files.

## How it works (high level)

The original dataset was built via this pipeline:
* Input: STL files from multiple miniature artists (multipart heroes, monsters, display pieces, etc.).
* Slicing: each STL is sliced in batch with PrusaSlicer using a consistent profile (CLI) with resin density of 1.1/g.
* Resin stats: UVtools inspects the .ctb/.sl1 files and exports resin stats, including resin usage in grams. This is the label.
* Geometry features: trimesh extracts mesh-level features: volume, surface area, implied mass, Euler characteristic, bounding box sizes, etc.
* Feature engineering: simple interaction & ratio features are added: volume × mass, surface/volume, surface/mass, bounding-box/volume, and similar.
* MiniRes is trained on this tabular data to predict UVtools’ “grams used” based only on these engineered features.

## Installation

Install the required dependencies:

```bash
pip install tensorflow xgboost pandas numpy huggingface_hub
```

Then add `minires.py` to your project (or install this repo as a package if you set it up that way).

## Basic usage

```python
import pandas as pd
from minires import minires

# Load your data
df = pd.read_csv("my_miniatures_features.csv")

# Create the model (downloads weights from Hugging Face on first run)
model = minires(verbose=1)

# Predict resin usage (grams)
y_pred = model.predict(df)

print(y_pred[:10])
```

`verbose=1` prints the ensemble weights and Keras progress. You can pass the full dataframe; the model will select the features it needs.

## Required features

The model expects a fixed set of feature columns (the same ones used in training).

You can inspect them at runtime:
```python
model = minires()
print(model.features)
```

Your dataframe must contain at least these columns. Extra columns are ignored. If any required columns are missing, `minires` will raise an error listing them.

Example single-row usage:
```python
import pandas as pd
from minires import minires

model = minires()

row = {
    "kb": 123.4,
    "volume": 56.7,
    "surface_area": 1234.5,
    "bbox_area": 789.1
    "euler_number": -1,
    "scale": 76.76
    "surface_volume_ratio": 0.8,
}

df_single = pd.DataFrame([row])
pred = model.predict(df_single)[0]

print("Predicted grams:", pred)
```

## Notes

The model approximates UVtools resin usage based on geometry/feature data.

Make sure you compute the same features as listed in `model.features`.

Weights are cached locally by `huggingface_hub` after the first download.

[MIT License](LICENSE)