# Crop Type Classification Using Remote Sensing & Machine Learning

This repository presents a complete pipeline for **crop type classification** using **remote sensing data** and **machine learning**, built on top of Google Earth Engine (GEE), TensorFlow, and scikit-learn. The project compares traditional ML methods and deep learning models using both **mono-temporal** and **multi-temporal** Sentinel-2 datasets for agricultural monitoring.

---

## Tech Stack

- **Python**
- **Google Earth Engine**
- **TensorFlow / Keras**
- **scikit-learn**
- **geemap & Folium**
- **Pandas / NumPy**
- **Matplotlib / Seaborn**

---

## Project Overview

Accurate crop type mapping is essential for **agricultural monitoring**, **yield estimation**, and **food security analysis**. This project uses **Sentinel-2 Harmonized surface reflectance** imagery and **USDA Cropland Data Layer (CDL)** labels to classify crops in two regions of the U.S.:

- **Easy AOI:** ~4 major crops (Eastern Arkansas)  
- **Hard AOI:** >20 crop classes (Eastern North Carolina)

Each area is evaluated using two data configurations:

- **Mono-temporal** (single timestamp per point)  
- **Multi-temporal** (full time series across the growing season)

Four models are compared:

- **LSTM**
- **CNN**
- **Random Forest**
- **SVM**

---

## Data Sources & Features

### **Satellite Data**
- Sentinel-2 Harmonized (S2-HARM) via GEE  
- Bands used:
  - B2 (Blue)
  - B3 (Green)
  - B4 (Red)
  - B8 (NIR)
- Computed index:  
  - **NDVI = (B8 − B4) / (B8 + B4)**

### **Ground Truth**
- USDA NASS Cropland Data Layer (CDL), 2024  
- 150 points sampled per class in each AOI  
- Non-agricultural classes filtered out

---

## Pipeline Summary

### 1. **AOI Selection**
Two rectangular AOIs defined to represent simple and complex crop diversity.

### 2. **Label Sampling**
Balanced sampling per class (150 points) using CDL.

### 3. **Feature Extraction (GEE)**
- Spectral bands + NDVI
- Cloud masking to remove contaminated pixels
- Exported as mono-temporal and multi-temporal datasets

### 4. **Dataset Construction**
- Mono-temporal: mid-season observation  
- Multi-temporal: full growing season  
  - Remove points with <10 timestamps  
  - Clean duplicates  
  - Export `.csv` and `.xlsx`

### 5. **Preprocessing**
- Class mapping (CDL code → crop names)
- Scaling & normalization
- Sequence padding (for LSTM)

### 6. **Modeling**
| Model | Mono-Temporal | Multi-Temporal |
|-------|----------------|----------------|
| LSTM | ❌ | ✔ Best performer |
| CNN | ✔ | ✔ |
| RF | ✔ Fastest | ✔ |
| SVM | ✔ | ✔ (lower accuracy) |

All models trained using 80/20 split and tuned via grid search.

### 7**Visualization**
Interactive maps and charts via **Folium** and **geemap** are provided for:
- NDVI time series  
- Class distribution  
- Sample point maps  
- Timestamp statistics  

---

## Key Results

### **Performance Summary**
| Model | Easy Mono | Easy Multi | Hard Mono | Hard Multi |
|-------|-----------|-------------|------------|-------------|
| **LSTM** | – | **0.94** | – | **0.88** |
| **CNN** | 0.89 | 0.91 | 0.76 | 0.83 |
| **Random Forest** | **0.91** | 0.87 | 0.78 | 0.80 |
| **SVM** | 0.87 | 0.84 | 0.72 | 0.76 |
---

## Insights & Discussion

- **LSTM is the top performer** when temporal data is available.
- **Random Forest is the best fast baseline model**, especially for mono-temporal data.
- Multi-temporal data provides **significant gains**, especially for crops with similar spectral signatures.
- Classification difficulty increases with number of crop classes (hard AOI).

- Temporal patterns (phenology) are **crucial** for distinguishing crops like corn & soybean.
- Class imbalance and spectral similarity remain major challenges in high-diversity regions.
- RF and SVM are efficient but plateau with complex, multi-temporal data.
- Visualizations were essential for understanding crop cycles and dataset quality.

---

## Limitations
- 150 samples per class may be insufficient for rare crops.
- Cloud masking may still leave noise.
- Mono-temporal snapshot may not represent all classes equally well.

---

## Future Work
- Expand AOIs and enlarge sample sizes
- Add Sentinel-1 (SAR) data for cloud-resistant modeling  
- Explore Transformer architectures for temporal modeling  
- Develop an interactive decision-support dashboard
