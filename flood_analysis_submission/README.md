# Flood Dynamics in South Sudan — Reproducible Analysis

**Course:** JBG060 Capstone Data Challenge · Eindhoven University of Technology & Zero Hunger Lab
**Authors:** Group 22
**Language:** Python 3.12 · Jupyter Notebook

---

## 1. Overview

Since 2019–2020, seasonal flooding in South Sudan has turned into multi-year extreme flooding that has displaced people, drowned livestock and deepened food insecurity. This project analyses that regime shift and supports anticipatory humanitarian action.

All code is in a single notebook, `reproducible_flood_analysis.ipynb`. Starting from the raw course data only, it:

1. **Builds all analysis datasets** from satellite flood masks, rainfall/runoff reanalysis, evapotranspiration, lake levels, river discharge, population, livestock, cropland, health facilities and IPC food-insecurity data (Section 3).
2. **Explores the flood regime shift** — the 2003–2019 baseline versus 2020–2025, and the 408 km² extreme-flood threshold (Section 4).
3. **Maps exposure** of the 512 payams (ADM3) to floods: population, cattle, cropland and health facilities (Section 4).
4. **Trains a Random Forest** that identifies high-flood-hazard payams (Section 5).
5. **Trains a GRU neural network** that gives early warning of extreme flood weeks 2–26 weeks ahead from upstream lake levels and climate drivers (Section 6).
6. **Reproduces the poster figures** for SQ1 (upstream drivers) and SQ3 (payam priority map for anticipatory action) (Section 7).

No preprocessed files are needed. Every intermediate dataset is computed in memory each time the notebook runs.

---

## 2. What you need

| Requirement | Details |
|:---|:---|
| Computer | Windows, macOS or Linux; at least 8 GB RAM; about 6 GB free disk space |
| Python | Version 3.12 (tested with 3.12.4). Python 3.10–3.12 should also work. |
| Raw data | The course data package `data-JBG060-2026` (≈ 4.5 GB, from the course Canvas page) |
| Run time | About 2 minutes on a recent laptop |
| Internet | Only needed to install the packages |

### Python packages (versions tested)

| Package | Version | Used for |
|:---|:---|:---|
| numpy | 1.26.4 | numerical arrays |
| pandas | 3.0.3 | tables and time series |
| geopandas | 1.1.4 | administrative boundaries, spatial joins |
| shapely | 2.1.2 | geometries |
| pyogrio | 0.13.0 | reading GeoJSON files |
| pyproj | 3.7.2 | coordinate systems |
| rasterio | 1.5.0 | reading GeoTIFF rasters |
| rasterstats | 0.21.0 | totals per payam (population, cattle, cropland) |
| affine | 2.4.0 | raster grid geometry |
| xarray | 2026.4.0 | reading ERA5 and lake NetCDF files |
| netCDF4 | 1.7.4 | reading NetCDF files |
| pyarrow | 24.0.0 | reading flood-mask parquet files |
| openpyxl | 3.1.5 | reading IPC Excel files |
| matplotlib | 3.10.9 | figures and maps |
| seaborn | 0.13.2 | heatmaps |
| scipy | 1.17.1 | statistics |
| scikit-learn | 1.9.1 | Random Forest and evaluation metrics |
| joblib | 1.6.0 | saving the trained model |
| torch (PyTorch) | 2.10.0 | GRU neural network (runs on CPU; no GPU needed) |
| jupyter | 1.0.0 | opening and running the notebook |

---

## 3. Folder structure

Put the files in **one folder** laid out like this:

```
Group22_submission/                      ← any name you like
├── README.md                            ← this file
├── reproducible_flood_analysis.ipynb    ← the entire analysis
└── data-JBG060-2026/                    ← raw data (you add this; see below)
    ├── Administrative boundaries/       ssd_admin0.geojson, ssd_admin3.geojson (country, payams)
    ├── worldpop/                        ssd_pop_2024_CN_100m_R2025A_v1.tif (population)
    ├── farmland/                        geonode__cattle_gha.tif, asap_mask_crop_v04.tif
    ├── health facilities/               Sub-Saharan_public_health_facilities.geojson
    ├── IPC/                             IPC_SS_Acute_Food_Insecurity_*.xlsx
    ├── flood_masks/                     compact_unusual/*.parquet, compact_recurring/*.parquet
    ├── rainfall and runoff/             ERA5_2000.nc … ERA5_2025.nc
    ├── evapotranspiration/              ET_2000/ … ET_2025/ (daily AgERA5 .nc files)
    ├── Water levels lakes/              water_level_victoria.txt, water_level_Kyoga.txt,
    │                                    water_level_altimetry_Albert.nc
    └── Darthmouth Flood Observatory/    100205_discharge.csv, 1505_discharge.csv
```

**The only folder you need to create is `data-JBG060-2026`.** Unzip the course data package and keep its sub-folder names exactly as they are. The `GDP` folder in the package is not used and can be left out.

The notebook also finds the raw data if the folder is called `raw_data`, or if it sits one level above the notebook.

### Outputs (created automatically)

When the notebook runs, it creates an `outputs/` folder next to itself. You do not need to create it.

| Folder | Contents |
|:---|:---|
| `outputs/tables/` | CSV results: payam exposure table, input coverage audit, two-factor exposure, action tiers, Random Forest predictions and CV results, 16-day risk forecast, GRU test metrics, SQ3 priority index |
| `outputs/figures/` | PNG figures: hydroclimatic regime and correlations, two-factor scatter, top-10 risk payams, health facilities vs. flooding, Random Forest evaluation, forecast heatmap and timeline, side-by-side risk maps |
| `outputs/poster/` | The poster figures `sq1_combined.png` and `sq3_priority_map.png` |
| `outputs/models/` | The trained models `random_forest_model.joblib` and `flood_early_warning_gru.pt` |
| `outputs/` (top level) | Copies of several of the files above, kept for backward compatibility |

The submitted notebook already shows all its outputs, so you can read the results without running anything.

---

## 4. Installation (step by step, no coding experience needed)

### Step 1 — Install Anaconda (includes Python and Jupyter)
1. Go to <https://www.anaconda.com/download> and download the installer for your system.
2. Run the installer and accept the default options.

### Step 2 — Open a terminal
- **Windows:** open the Start menu, search for **Anaconda Prompt** and open it.
- **macOS:** open **Terminal** (press `Cmd + Space`, type *Terminal*, press Enter).

### Step 3 — Create a clean environment (recommended)
Type each line below into the terminal and press Enter after each one:

```
conda create -n flood python=3.12 -y
conda activate flood
```

### Step 4 — Install the packages
Copy this whole command into the terminal and press Enter. It can take a few minutes.

```
pip install numpy==1.26.4 pandas==3.0.3 geopandas==1.1.4 shapely==2.1.2 pyogrio==0.13.0 pyproj==3.7.2 rasterio==1.5.0 rasterstats==0.21.0 affine==2.4.0 xarray==2026.4.0 netCDF4==1.7.4 pyarrow==24.0.0 openpyxl==3.1.5 matplotlib==3.10.9 seaborn==0.13.2 scipy==1.17.1 scikit-learn==1.9.1 joblib==1.6.0 torch==2.10.0 jupyter==1.0.0
```

---

## 5. Running the notebook

1. Open the terminal again (Step 2). If you created the environment, activate it first with `conda activate flood`.
2. Start Jupyter:
   ```
   jupyter notebook
   ```
   A page opens in your web browser showing the folders on your computer.
3. In that page, click through the folders until you reach your submission folder, then click **`reproducible_flood_analysis.ipynb`** to open it.
4. In the menu at the top, choose **Kernel → Restart Kernel and Run All Cells…** and confirm.
   (In older Jupyter versions this is **Kernel → Restart & Run All**.)
5. Wait about 2 minutes. While a cell is running it shows `[*]` on its left; when it finishes, the `*` becomes a number. Results and figures appear below each cell.
6. The run is complete when the last cell prints `Pipeline audit finished: 18/18 core deliverables verified.`
7. All tables, figures and models are now in the `outputs/` folder (see Section 3).

**Using VS Code instead:** open the submission folder in VS Code, open the notebook, choose the `flood` Python environment as the kernel (top right), then click **Restart** and **Run All**.

---

## 6. Troubleshooting & FAQ

| Problem | Solution |
|:---|:---|
| `FileNotFoundError: Raw data package not found` | The `data-JBG060-2026` folder is missing or in the wrong place. Put it in the same folder as the notebook, and check that it contains `Administrative boundaries`. |
| `FileNotFoundError: Required raw input missing: …` | A raw file or sub-folder is missing or has been renamed. The error message shows the exact path the notebook expected. Restore the original folder and file names from the data package. |
| `ModuleNotFoundError: No module named '…'` | The packages are not installed in the environment Jupyter is using. Run `conda activate flood` before `jupyter notebook`, or in VS Code select the `flood` kernel. Then repeat Step 4. |
| Installing `rasterio`, `netCDF4` or `geopandas` with pip fails (most often on Windows) | Install them with conda instead: `conda install -c conda-forge geopandas rasterio rasterstats netcdf4 xarray`, then run the pip command again. |
| The kernel dies or restarts while Section 3 is running | The computer ran out of memory. Close other programs and try again; at least 8 GB RAM is needed. |
| Results are slightly different from the poster | Training is seeded (`GLOBAL_SEED = 42`), but tiny numerical differences between operating systems or package versions can change the last decimals of model scores. |
| Some payams in far-northern Upper Nile have no flood data | Expected: the satellite flood masks provided only cover 0–10°N (see Section 8). |

---

## 7. Credits & Acknowledgements

**Data sources** (provided in the JBG060 course data package):
- Administrative boundaries of South Sudan (ADM0–ADM3) — OCHA / Humanitarian Data Exchange: <https://data.humdata.org>
- Daily flood masks — NASA MODIS/VIIRS Global Flood Product (MCDWD): <https://www.earthdata.nasa.gov/global-flood-product>
- ERA5 hourly precipitation and runoff — Copernicus Climate Change Service: <https://cds.climate.copernicus.eu>
- AgERA5 reference evapotranspiration — Copernicus / Wageningen Environmental Research: <https://doi.org/10.24381/cds.6c68c9bb>
- Lake Albert water level — DAHITI, DGFI-TUM: <https://dahiti.dgfi.tum.de>
- Lake Victoria and Kyoga water levels — USDA G-REALM: <https://ipad.fas.usda.gov/cropexplorer/global_reservoir/>
- River discharge — Dartmouth Flood Observatory River Watch: <https://floodobservatory.colorado.edu>
- Population 2024 — WorldPop: <https://www.worldpop.org>
- Cattle density (GHA region) and ASAP cropland mask — ICPAC / JRC ASAP: <https://agricultural-production-hotspots.ec.europa.eu>
- Public health facilities in sub-Saharan Africa — Maina et al. (2019), WHO / KEMRI-Wellcome
- Acute food insecurity — Integrated Food Security Phase Classification (IPC): <https://www.ipcinfo.org>

**Software:** NumPy, pandas, GeoPandas, Shapely, rasterio, rasterstats, xarray, netCDF4, PyArrow, Matplotlib, seaborn, SciPy, scikit-learn and PyTorch — all open source; see the package table in Section 2.

---

## 8. Limitations, Legal & Ethical Considerations

- **No personal data.** All inputs are public, aggregated datasets (gridded rasters, county-level IPC estimates, facility locations). No individual-level information is processed.
- **Flood mask coverage.** The flood masks cover two MODIS tiles (0–10°N), so the northernmost part of Upper Nile has no flood observations. Cloud cover can also hide flooding on some days. Hazard in those areas is therefore underestimated.
- **Simulated weather forecast (Section 5.1).** The 16-day rainfall forecast is generated synthetically with a fixed random seed. It shows how the structural risk and rainfall would be combined; it is **not** a live weather forecast and must not be used for operational decisions.
- **Model limitations.** The Random Forest learns where flooding occurred in 2022; the GRU is trained on 2003–2018 and tested on 2019–2025. Early-warning skill drops sharply beyond about 4 weeks of lead time (see `gru_forward_test_metrics.csv`). Both models describe statistical associations, not causes.
- **Fixed analyst choices.** The flood regions (Sudd = Jonglei, Unity, Lakes; West = Warrap and Northern & Western Bahr el Ghazal; East/Sobat = Upper Nile), the rainfall boxes and the priority-index weights are choices made by the analysts and are documented in the notebook. Other choices could change the rankings.
- **Responsible use.** The priority map ranks payams for *anticipatory* action. It is meant to support humanitarian experts, not to replace local knowledge, field assessments or the judgement of affected communities. A payam with a low priority score is not necessarily safe.
- **Data licences.** Each dataset remains subject to its provider's licence and terms of use. The code is provided for educational and research purposes.
