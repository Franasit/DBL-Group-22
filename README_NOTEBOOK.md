# Reproducible Master Notebook: Flood Dynamics in South Sudan

**Course**: JBG060 Capstone Data Challenge · Eindhoven University of Technology & Zero Hunger Lab  
**Group**: Group 22  
**Target Notebook**: [`reproducible_flood_analysis.ipynb`](reproducible_flood_analysis.ipynb)  
**Deliverable Type**: Self-Contained End-to-End Scientific Notebook & Execution Guide

---

## 1. Introduction and Overview

### 1.1 Purpose of the Notebook

This Jupyter notebook (`reproducible_flood_analysis.ipynb`) implements the entire data science, geospatial, machine learning, and deep learning workflow for our project: **"From seasonal to extreme flooding in South Sudan: drivers, upstream early warning and ADM3 prioritisation."**

Executing this single notebook from top to bottom executes the entire analysis and directly reproduces **every result, figure, map, and table** presented in our scientific poster and technical report. The notebook is fully self-contained and decoupled from the broader development codebase: all necessary datasets are provided in the accompanying `data/` directory, and all outputs are automatically saved to `outputs/`.

### 1.2 What the Analysis Accomplishes

The notebook answers three interconnected scientific and operational questions:

1. **Upstream Early Warning (SQ1)**: Analyzes equatorial lake storage anomalies (Lake Albert, Lake Victoria, Lake Kyoga) and White Nile discharge to identify the hydroclimatic drivers behind South Sudan's unprecedented post-2020 flood regime shift. It demonstrates that Lake Albert anomalies surge extreme flood odds by **3.83-fold** (95% CI: [2.22, 10.33]).
2. **Predictive Machine Learning (SQ2 / Random Forest)**: Standardizes socio-economic, agricultural, and demographic indicators across all **512 ADM3 Payams**, performing Stratified 5-Fold Cross-Validation and `GridSearchCV` to predict high flood-hazard payams based on observable pre-flood features.
3. **Deep Learning Sequence Modeling (GRU Early Warning)**: Trains a multi-horizon PyTorch Gated Recurrent Unit (`nn.GRU`) network consuming 52-week lookback tensors to forecast extreme flood probabilities across 5 lead times (**2, 4, 8, 13, and 26 weeks ahead**), benchmarked against Climatology, Autoregressive Persistence, Logistic Regression, and Histogram Gradient Boosting.
4. **Anticipatory Humanitarian Prioritisation (SQ3)**: Combines flood hazard ($H$), population/cattle exposure ($E$), and acute food insecurity vulnerability ($V$, IPC Phase 3+) into an INFORM-style geometric index:
   $$\text{Priority} = \left(H \times E \times V\right)^{1/3}$$
   ranking the top 10% highest-priority payams (Rubkona, Fangak, Ayod, Duk, Twic East) for anticipatory cash and levee deployments.

---

## 2. Requirements and Exact Dependency Versions

### 2.1 System Prerequisites

- **Operating System**: macOS (Intel or Apple Silicon), Linux (Ubuntu, Debian, CentOS), or Windows 10/11 (PowerShell or WSL).
- **Python Runtime**: **Python 3.12** is recommended and tested (`3.12.4`). Compatible with Python $\ge 3.10$ and $< 3.13$.
- **Hardware**:
  - Memory: 8 GB RAM minimum (16 GB recommended for seamless geospatial joins).
  - Processor: 4+ CPU cores. GPU (CUDA or Apple MPS) is supported but not required; full CPU execution takes under 2 minutes.
  - Disk Space: ~150 MB for the code, bundled data, and generated artifacts.

### 2.2 Complete Dependency Matrix

The following table details **every package required to run the notebook**, its tested version, compatible range, and exact purpose within the analysis:

| Package Name          | Tested Version |   Compatible Range   | Purpose in Notebook                                                                               |
| :-------------------- | :------------: | :------------------: | :------------------------------------------------------------------------------------------------ |
| **`python`**          |    `3.12.4`    |  `>= 3.10, < 3.13`   | Underlying Python runtime environment                                                             |
| **`numpy`**           |    `1.26.4`    | `>= 1.24.0, < 2.0.0` | High-performance tensor & matrix operations (kept at `< 2.0` for PyTorch / C-extension stability) |
| **`pandas`**          |    `3.0.3`     |      `>= 2.0.0`      | Tabular data manipulation, time-series resampling (`W-SUN`), weekly anomaly calculation           |
| **`geopandas`**       |    `1.1.4`     |     `>= 0.14.0`      | Geospatial vector processing, spatial joins (`sjoin`), payam boundary Dissolve operations         |
| **`shapely`**         |    `2.1.2`     |      `>= 2.0.0`      | Geometric spatial primitives (polygons, points, spatial intersections, point-in-polygon checks)   |
| **`rasterio`**        |    `1.5.0`     |      `>= 1.3.0`      | Geospatial raster I/O and mask extraction                                                         |
| **`scikit-learn`**    |    `1.9.1`     |      `>= 1.3.0`      | Random Forest, GridSearchCV, Stratified K-Fold CV, ROC-AUC, Confusion Matrix, StandardScaler      |
| **`scipy`**           |    `1.17.1`    |     `>= 1.11.0`      | Statistical calculations, Spearman rank correlations, spatial interpolation                       |
| **`torch` (PyTorch)** |    `2.2.2`     |      `>= 2.0.0`      | Deep learning framework: multi-horizon `nn.GRU`, TensorDataset, DataLoader, BCEWithLogitsLoss     |
| **`matplotlib`**      |    `3.10.9`    |      `>= 3.7.0`      | Scientific visualization engine: multi-panel figures, poster layouts, maps, time-series charts    |
| **`seaborn`**         |    `0.13.2`    |     `>= 0.12.0`      | Enhanced statistical plotting (driver cross-correlation heatmaps)                                 |
| **`joblib`**          |    `1.5.3`     |      `>= 1.3.0`      | Model serialization and disk persistence for trained scikit-learn models (`.joblib`)              |
| **`pyarrow`**         |    `24.0.0`    |     `>= 14.0.0`      | High-throughput columnar Parquet I/O for 23-year daily hydroclimatic series                       |
| **`jupyter`**         |    `1.0.0`     |      `>= 1.0.0`      | Core Jupyter metapackage                                                                          |
| **`ipykernel`**       |    `6.28.0`    |     `>= 6.20.0`      | Jupyter execution kernel for Python 3 environments                                                |
| **`nbconvert`**       |    `7.10.0`    |      `>= 7.0.0`      | Programmatic headless execution and export of Jupyter notebooks                                   |
| **`nbformat`**        |    `5.9.2`     |      `>= 5.8.0`      | Specification parser for notebook cells, metadata, and compliant cell IDs                         |

All dependencies are pinned in [`requirements.txt`](requirements.txt).

---

## 3. Installation Instructions

Follow these step-by-step instructions to prepare a clean environment on your computer.

### Step 1: Open Terminal / Command Prompt and Navigate to the Directory

Ensure you are in the submission folder where `reproducible_flood_analysis.ipynb` and `requirements.txt` are located:

```bash
cd flood_analysis_submission
```

### Step 2: Create a Virtual Environment

- **macOS / Linux**:

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

- **Windows (PowerShell)**:

  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```

- **Windows (Command Prompt)**:

  ```cmd
  python -m venv .venv
  .\.venv\Scripts\activate.bat
  ```

- **Alternative: Using Conda**:
  ```bash
  conda create -n flood_analysis python=3.12 -y
  conda activate flood_analysis
  ```

### Step 3: Install Required Dependencies

Upgrade `pip` to the latest version and install the verified dependency versions:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 4: Register the Environment with Jupyter

Register your virtual environment so it appears cleanly in Jupyter's kernel dropdown:

```bash
python -m ipykernel install --user --name=flood_analysis --display-name="Python (Flood Analysis)"
```

---

## 4. Usage: How to Run the Notebook

You can run the notebook either **interactively** (recommended for exploring visualizations) or **headlessly** via the command line.

### Option A: Interactive Execution in Jupyter (Recommended)

1. **Launch Jupyter Lab or Notebook**:

   ```bash
   jupyter lab reproducible_flood_analysis.ipynb
   # or
   jupyter notebook reproducible_flood_analysis.ipynb
   ```

   _(If using VS Code, open the folder in VS Code, open `reproducible_flood_analysis.ipynb`, and select the `.venv` interpreter in the top right)._

2. **Select Kernel**:
   Confirm that the active kernel is **Python (Flood Analysis)**.

3. **Execute All Cells**:
   In the top menu, click **Kernel > Restart Kernel and Run All Cells** (or press `Shift + Enter` to step through cell by cell).

4. **Inspect Generated Visualizations & Outputs**:
   All figures will display inline directly below their respective code cells and will automatically be saved into the `outputs/` directory.

---

### Option B: Command-Line / Headless Execution (Fast Batch Mode)

To run the entire notebook from start to finish without opening a browser and verify that every calculation completes with zero errors:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace reproducible_flood_analysis.ipynb
```

- Expected runtime: **~90 to 120 seconds** on standard hardware.
- Return code: `0` confirms 100% successful execution across all 20 cells.

---

## 5. Notebook Section-by-Section Architecture

The notebook is divided into 8 logical, self-contained sections:

```text
reproducible_flood_analysis.ipynb
├── Section 1: Executive Overview & Problem Background
├── Section 2: Environment Setup, Reproducibility & Helper Utilities
├── Section 3: Data Ingestion & Spatial Data Cleaning
├── Section 4: Exploratory Data Analysis (EDA) & Feature Engineering
│   ├── 4.1 Basin-Scale Hydroclimatic Dynamics & Regime Shift
│   └── 4.2 ADM3 Spatial Exposure & Two-Factor Risk Decomposition
├── Section 5: Machine Learning Modeling: Random Forest Classification
├── Section 6: Deep Learning Modeling: Multi-Horizon GRU Early Warning System
├── Section 7: Reproduction of Poster & Report Deliverables
│   ├── 7.1 Sub-Question 1 (SQ1): Upstream Drivers & Regime Shift Poster Graphics
│   └── 7.2 Sub-Question 3 (SQ3): ADM3 Humanitarian Prioritisation Map
└── Section 8: Pipeline Integrity Verification & Summary
```

### Summary of What Each Section Does:

- **Section 2 (`Setup`)**: Configures thread safety (`OMP_NUM_THREADS=1` to prevent OpenMP crashes between PyTorch and Scikit-Learn), sets random seeds (`GLOBAL_SEED = 42`), and anchors paths.
- **Section 3 (`Data Cleaning`)**: Loads `ssd_admin3.geojson` and `adm3_exposure_aligned.csv`, validates CRS (`EPSG:4326`), verifies geometric validity, and prints a missing value audit across all 512 payams.
- **Section 4.1 (`Hydroclimatic EDA`)**: Aggregates daily satellite flood masks and meteorological fluxes to weekly series, establishes the 2003–2019 baseline vs. 2020–2025 extreme flood regimes, derives the $408\,\text{km}^2$ extreme flood threshold, and plots the seasonal flood cycle and cross-driver correlation heatmap.
- **Section 4.2 (`Two-Factor Exposure`)**: Normalizes population, livestock, cropland, and health facilities; creates Factor 1 (Human/Pastoral) and Factor 2 (Agronomic/Infrastructure) risk composites; and plots the risk divergence scatter plot.
- **Section 5 (`Random Forest`)**: Formulates the high flood-hazard classification target ($\ge 75\text{th}$ percentile threshold), optimizes hyperparameters using Stratified 5-Fold CV and `GridSearchCV`, evaluates test-set ROC-AUC and Average Precision, plots feature importances and ROC curves, and scores all 512 payams into operational risk tiers.
- **Section 6 (`GRU Early Warning`)**: Constructs a 52-week sequence lookback tensor ($X \in \mathbb{R}^{N \times 52 \times 6}$), builds the PyTorch `FloodEarlyWarningGRU` architecture, trains across multi-horizon targets ($h \in \{2, 4, 8, 13, 26\}$ weeks), evaluates forward regime-shift test metrics, and displays the multi-model Leave-One-Year-Out (LOYO) benchmark table.
- **Section 7.1 (`SQ1 Poster Graphics`)**: Replicates `sq1_combined.png` displaying the White Nile basin map, flooded land in 2023, regional flood extent time series (Sudd vs. West vs. East), and Lake Albert water level anomalies.
- **Section 7.2 (`SQ3 Priority Map`)**: Calculates the INFORM geometric priority index across 512 payams, identifies the top 10% highest-priority hotspots (Rubkona, Fangak, Ayod, Duk, Twic East), and renders `sq3_priority_map.png` with Hazard, Exposure, and Vulnerability insets.
- **Section 8 (`Integrity Verification`)**: Executes automated assertion checks confirming all 11 core tables, models, and figures have been generated and match expected file sizes.

---

## 6. Input Data Directory (`data/`)

All necessary input files are bundled inside the `data/` directory so the notebook does not depend on external downloads or large raster directories:

| File Name                              | File Size | Description                                                                                                             |
| :------------------------------------- | :-------: | :---------------------------------------------------------------------------------------------------------------------- |
| **`ssd_admin3.geojson`**               |  3.6 MB   | Vector polygon geometries for all 512 ADM3 Payams in South Sudan                                                        |
| **`ssd_admin0.geojson`**               |  778 KB   | Vector boundary for the national border of South Sudan                                                                  |
| **`adm3_exposure_aligned.csv`**        |   98 KB   | Zonal statistics & counts (population, cattle, cropland, health clinics, 2022 flood detections)                         |
| **`county_ipc_flood_analysis.csv`**    |   20 KB   | County-level acute food insecurity metrics (IPC Phase 3+ population shares)                                             |
| **`daily_series.parquet`**             |  1.1 MB   | 23-year daily time series (2003–2025): satellite flood pixels, lake levels, Nile discharge, ERA5 rain/runoff, AgERA5 ET |
| **`_flood_grid_monthly_frac.npz`**     |  1.1 MB   | 0.04° monthly gridded flooded fraction cache (recurring vs. unusual flood masks)                                        |
| **`_flood_regions.parquet`**           |  177 KB   | Regional satellite flood series (Sudd wetland, West region, East region)                                                |
| **`external_lakes_grealm.parquet`**    |   19 KB   | NASA G-REALM satellite altimetry for Lakes Victoria and Kyoga (1992–2026)                                               |
| **`external_climate_indices.parquet`** |   30 KB   | NOAA Indian Ocean Dipole (IOD) Dipole Mode Index series                                                                 |

---

## 7. Generated Outputs Directory (`outputs/`)

Running the notebook populates the `outputs/` folder with the following deliverables:

```text
outputs/
├── tables/
│   ├── adm3_two_factor_exposure_analysis.csv   # Normalized indicators and 2-factor risk scores
│   ├── adm3_random_forest_predictions.csv      # Model-scored flood probability and risk tiers for 512 payams
│   ├── gru_forward_test_metrics.csv            # GRU evaluation metrics (AUC, PR-AUC, Brier) across 5 lead times
│   └── sq3_adm3_priority_index.csv             # INFORM geometric priority index and national rankings (1-512)
├── figures/
│   ├── eda_hydroclimatic_regime_and_correlations.png # Seasonal flood cycles and driver correlation matrix
│   ├── two_factor_risk_scatter.png             # Divergence scatter plot between Factor 1 and Factor 2
│   └── random_forest_evaluation.png            # Feature importance ranking and test-set ROC curve
├── poster/
│   ├── sq1_combined.png                        # SQ1 poster figure: Basin map + regional flood & lake level series
│   └── sq3_priority_map.png                    # SQ3 poster figure: National priority map with H, E, V insets
└── models/
    ├── random_forest_model.joblib              # Tuned Random Forest classifier artifact
    └── flood_early_warning_gru.pt              # PyTorch state dictionary for trained GRU network
```

---

## 8. Troubleshooting and FAQs

### Q1: I encounter an OpenMP error (`OMP: Error #15: Initializing libiomp5.dylib...`) on macOS.

**Cause**: PyTorch and Scikit-Learn ship separate OpenMP runtimes that collide when multi-threaded in the same process on macOS.  
**Fix**: Cell 2 of the notebook explicitly sets `os.environ["OMP_NUM_THREADS"] = "1"` before importing PyTorch or Scikit-Learn. If executing outside the notebook, run `export OMP_NUM_THREADS=1` in your terminal first.

### Q2: Why does `requirements.txt` specify `numpy<2`?

**Cause**: NumPy 2.0 introduced ABI breaking changes. Certain compiled C-extensions (older PyTorch binaries and numexpr/bottleneck) may throw `ImportError: _ARRAY_API not found` under NumPy 2.x.  
**Fix**: Using `numpy==1.26.4` guarantees complete stability across all platforms.

### Q3: How long does the notebook take to execute?

- Cells 1–5 (Data cleaning, EDA, Two-Factor Exposure): ~5 seconds.
- Cell 6 (Random Forest GridSearchCV with 5 folds): ~10 seconds.
- Cells 7–8 (PyTorch GRU sequence construction and training): ~25 seconds.
- Cell 9 (Poster SQ1 & SQ3 spatial rendering with 45,000 grid points): ~40 seconds.
- **Total execution time**: ~**90 seconds**.

### Q4: Can I run this notebook on Windows?

**Yes**. The code uses Python's standard `pathlib.Path` for all file system operations, ensuring forward/backward slash compatibility across Windows, macOS, and Linux.

---

## 9. Credits and Acknowledgements

This research was conducted as part of the **JBG060 Capstone Data Challenge (2026)** at **Eindhoven University of Technology** in collaboration with the **Zero Hunger Lab (ZHL)**.

### Datasets and Organizations:

- **NASA Global Flood Product (MCDWD)**: Optical flood inundation mapping (MODIS/VIIRS).
- **ECMWF**: ERA5 atmospheric reanalysis precipitation and surface runoff; AgERA5 reference evapotranspiration.
- **DAHITI & NASA G-REALM**: Satellite altimetry records for Lake Albert, Lake Victoria, and Lake Kyoga.
- **Dartmouth Flood Observatory (DFO)**: River discharge records at Malakal.
- **WorldPop & ICPAC**: High-resolution population surfaces and East Africa livestock headcount.
- **Joint Research Centre (JRC)**: Anomaly Hotspots of Agricultural Production (ASAP) cropland masks.
- **Integrated Food Security Phase Classification (IPC)**: Acute food insecurity classifications.

---

## 10. Legal, Privacy, and Ethical Considerations

1. **Non-Identifiable Data**: All exposure datasets represent aggregated spatial summaries or 100m grid cell approximations. No personally identifiable information (PII) is included or accessible.
2. **Humanitarian Prioritisation Ethics**: Algorithmic rankings (such as the ADM3 Priority Score) are designed as **decision-support tools** to inform early warning and anticipatory financing. They must never replace ground-truth field verification, community engagement, or consideration of local conflict dynamics.
3. **Sensor Limitations**: Satellite optical sensors cannot penetrate cloud cover during peak convective rainfall events. While weekly maximum aggregation mitigates this limitation, flash pluvial floods in cloud-covered periods may have minor detection delays.
4. **License & Academic Use**: This code is provided for academic evaluation, peer review, and non-commercial humanitarian research.
