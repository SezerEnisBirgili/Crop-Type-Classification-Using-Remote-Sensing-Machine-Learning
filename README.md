# Crop Type Classification Using Remote Sensing & Machine Learning

This repository presents a complete pipeline for crop type classification using remote sensing data and machine learning, built on top of Google Earth Engine (GEE), TensorFlow, and scikit-learn. The project compares traditional ML methods and deep learning models using both mono-temporal and multi-temporal Sentinel-2 datasets for agricultural monitoring.

## Tech Stack

- Python
- Google Earth Engine
- TensorFlow / Keras
- scikit-learn
- geemap & Folium
- Pandas / NumPy
- Matplotlib / Seaborn

## Project Overview

Accurate crop type mapping is essential for agricultural monitoring, yield estimation, and food security analysis. This project uses Sentinel-2 Harmonized surface reflectance imagery and USDA Cropland Data Layer (CDL) labels to classify crops in two regions of the U.S.:

- **Easy AOI:** 4 dominant crop classes (Eastern Arkansas)
- **Hard AOI:** 20 crop classes (Eastern North Carolina)

Each area is evaluated using two data configurations:

- **Mono-temporal** (single observation per point, taken nearest to July 15, 2024)
- **Multi-temporal** (full time series across the growing season)

Four models are compared:

- LSTM
- CNN
- Random Forest
- SVM

## Data Sources & Features

**Satellite Data**
- Sentinel-2 Harmonized (S2-HARM) Surface Reflectance via GEE
- Bands used:
  - B2 (Blue)
  - B3 (Green)
  - B4 (Red)
  - B8 (NIR)
- Computed index:
  - NDVI = (B8 − B4) / (B8 + B4)
- Cloud-free observations only, via GEE's built-in `QA60` cloud mask

**Ground Truth**
- USDA NASS Cropland Data Layer (CDL), 2024
- Minimum 50 sampled locations per class for the mono-temporal dataset; minimum 1,000 timestamps per class for the multi-temporal dataset (roughly 50 locations per class at an average of 20 timestamps per location)
- Non-agricultural, non-crop, non-active-cropland, and double-crop classes filtered out (see Pipeline step 5)
- Resulting dataset sizes: ~32,000 timestamps across ~1,500 points for the easy AOI; ~168,000 timestamps across ~8,700 points for the hard AOI

## Pipeline Summary

**1. AOI Selection**
Two rectangular AOIs defined to represent simple and complex crop diversity.

**2. Label Sampling**
Locations sampled per class from CDL: a minimum of 50 locations per class for the mono-temporal dataset, and a minimum of 1,000 timestamps per class (~50 locations at ~20 timestamps average) for the multi-temporal dataset.

**3. Feature Extraction (GEE)**
- Spectral bands + NDVI
- Cloud masking to remove contaminated pixels
- Exported as separate mono-temporal and multi-temporal CSV datasets (`crops_easy.csv`, `crops_hard.csv`)

**4. Dataset Construction**
- Mono-temporal: single observation per location, nearest to July 15, 2024 (mid-season)
- Multi-temporal: full growing season, spanning March 10 – October 21, 2024
- Remove points with fewer than 10 valid timestamps
- Remove duplicate records sharing the same (longitude, latitude, date)

**5. Preprocessing**
- Class mapping (CDL code → crop name)
- Agricultural class filtering, removing four groups of irrelevant labels: non-agricultural/infrastructure classes, natural ecosystems and non-crop vegetation, non-active agricultural land and ambiguous forage, and double-crop categories
- Feature scaling and normalization
- Temporal alignment and sequence padding (for LSTM/CNN)

**6. Modeling**

| Model | Mono-Temporal | Multi-Temporal |
|---|---|---|
| LSTM | ✔ (weaker; heavily adapted for non-sequential input) | ✔ Best performer |
| CNN | ✔ | ✔ |
| RF | ✔ Fastest | ✔ |
| SVM | ✔ | ✔ (scales worse on complex data) |

All models trained on an 80/20 train-test split. LSTM and CNN were trained with a batch size of 32 for up to 80 epochs, though training typically stopped around epoch 20 once validation performance plateaued.

**7. Visualization**
Interactive maps and charts via Folium and geemap are provided for:
- NDVI time series by crop type
- Class distribution
- Sample point maps
- Timestamp statistics

## Insights & Discussion

Multi-temporal data vastly improves accuracy over mono-temporal data across every model, with a single exception: Random Forest on the easy dataset, where the gain is marginal (0.85 → 0.87) compared to double-digit-point gains everywhere else.

LSTM benefits most from multi-temporal information and is the strongest model on the hardest classification task (Hard Multi), gaining 0.51 in accuracy over its mono-temporal score and outperforming every other model on the 20-class region once sequential NDVI patterns are available.

Random Forest is the best fast baseline model, especially for mono-temporal data, and trains the fastest of the four.

Classification difficulty increases sharply with the number of crop classes (hard AOI), where class imbalance and spectral overlap are more pronounced.

Temporal patterns (crop phenology) are crucial for distinguishing spectrally similar crops, which is why LSTM's advantage grows precisely where classification is hardest.

RF and SVM are efficient but plateau on complex, multi-temporal data relative to the neural models.

Visualizations, including per-crop NDVI time series, were essential for confirming that distinguishable growth trajectories exist across crop types before modeling.

## Limitations

- The minimum of 50 sampled locations per class (~1,000 timestamp records per class in the multi-temporal setting) may be insufficient for rare crops, especially in the high-diversity hard AOI.
- Cloud masking, while robust, may still admit artifacts that degrade temporal signal quality.
- Mono-temporal data relies on a single mid-season date, which may not represent all crop phenologies equally.
- Only two AOIs were selected, so results cannot be generalized to a given crop class everywhere in the world; the same crop in a different climate or region may behave differently.
- Overall data volume is limited (~32k timestamps/~1.5k points for easy, ~168k timestamps/~8.7k points for hard); the easy dataset's small size relative to its low class count is suspected to encourage overfitting, while the hard dataset likely needs substantially more data or more robust models to close the gap.

## Future Work

- Explore enhanced sampling strategies to improve data quality and coverage
- Use larger datasets to test the accuracy limits of the current model configurations
- Investigate advanced sequence models, such as Transformers, to better capture temporal dependencies
