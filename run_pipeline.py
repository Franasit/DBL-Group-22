#!/usr/bin/env python3
"""
Master End-to-End Pipeline Runner: Flood Dynamics in South Sudan
================================================================
This script executes the entire data science workflow sequentially,
starting from raw datasets (spatial rasters, vectors, flood masks, and
hydrological records) up to predictive machine learning models and future
risk forecasting.

Workflow Stages:
----------------
  [Stage 1: Upstream Hydrological Analysis]
    - Long-term trend analysis of Lake Victoria, Kyoga, Albert, and Nile discharge.
    - Script: scripts/upstream_analysis/01_upstream_trends_and_anomalies.py

  [Stage 2: Spatial Data Cleaning & Multi-Source Alignment (ADM3 Payam Level)]
    - Zonal statistics extraction (WorldPop population, cattle counts, cropland masks)
    - Spatial join with public health facilities point data
    - Spatial join with 2022 satellite flood masks (MODIS/VIIRS)
    - Script: scripts/cleaning_data/align_exposure_to_adm3.py
    - Output: outputs/adm3_exposure_aligned.csv

  [Stage 3: Normalization & Exposure Index Engineering]
    - Min-max standardization of hazard and exposure indicators
    - Script: scripts/eda/normalize_exposure.py
    - Output: outputs/adm3_exposure_normalized.csv

  [Stage 4: Two-Factor Exposure Analysis (EDA)]
    - Factor 1: Human & Pastoral Capital (Population + Cattle)
    - Factor 2: Agronomic & Infrastructure Capital (Cropland + Health Facilities)
    - Maps, scatter plots, and risk index generation
    - Script: scripts/eda/analyze_two_exposure_factors.py
    - Output: outputs/adm3_two_factor_exposure_analysis.csv

  [Stage 5: Spatial Risk Validation on Health Infrastructure]
    - Overlay of health infrastructure against 2022 historical flood density
    - Script: scripts/eda/rq1_spatial_risk_map.py

  [Stage 6: Predictive Machine Learning (Random Forest Classification)]
    - Hyperparameter tuning via 5-fold Stratified GridSearchCV
    - Model evaluation (ROC-AUC, Precision, Recall, F1)
    - Feature importance & model persistence (.joblib)
    - Script: scripts/eda/prediction_random_forest.py
    - Outputs: outputs/random_forest_model.joblib, outputs/adm3_random_forest_predictions.csv

  [Stage 7: Near-Future Flood Forecasting (Trained RF Model + Live Open-Meteo Weather)]
    - Ingests real-time 16-day rainfall forecasts for 512 payam centroids
    - Combines structural vulnerability with daily pluvial triggers
    - Generates daily time series, heatmap, timeline, and risk maps
    - Script: scripts/eda/predict_future_floods.py
    - Outputs: outputs/future_flood_risk_daily.csv, outputs/future_flood_risk_summary.csv

Usage:
------
    python run_pipeline.py
    python run_pipeline.py --skip-weather  # To skip external API forecast calls
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def run_stage(step_num: int, title: str, script_rel_path: str, args: list = None):
    script_path = PROJECT_ROOT / script_rel_path
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    print("\n" + "=" * 78)
    print(f" [STAGE {step_num}] {title}")
    print(f" Executing: {script_rel_path}")
    print("=" * 78)

    cmd = [sys.executable, str(script_path)] + (args or [])
    start_time = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"\n[ERROR] Stage {step_num} failed with return code {result.returncode} after {elapsed:.1f}s.")
        sys.exit(result.returncode)
    else:
        print(f"[SUCCESS] Stage {step_num} completed in {elapsed:.1f}s.")


def main():
    parser = argparse.ArgumentParser(description="Run complete flood analysis & prediction pipeline.")
    parser.add_argument(
        "--skip-weather",
        action="store_true",
        help="Skip the live 16-day Open-Meteo weather forecast download (predict_future_floods.py).",
    )
    cli_args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.time()

    print("\n" + "#" * 78)
    print(" FLOOD DYNAMICS IN SOUTH SUDAN: MASTER REPRODUCIBLE PIPELINE")
    print(f" Project Root: {PROJECT_ROOT}")
    print(f" Output Directory: {OUTPUT_DIR}")
    print("#" * 78)

    # Stage 1: Upstream hydrology & trends
    run_stage(
        1,
        "Upstream Hydrological Analysis (Lakes Victoria, Kyoga, Albert & Discharge)",
        "scripts/upstream_analysis/01_upstream_trends_and_anomalies.py",
    )

    # Stage 2: Spatial data cleaning & alignment to ADM3 Payams
    run_stage(
        2,
        "Spatial Data Cleaning & Multi-Source Alignment (ADM3 Level)",
        "scripts/cleaning_data/align_exposure_to_adm3.py",
    )

    # Stage 3: Normalization & composite indices
    run_stage(
        3,
        "Indicator Normalization & Combined Exposure Score",
        "scripts/eda/normalize_exposure.py",
    )

    # Stage 4: Two-factor exposure analysis & risk composites
    run_stage(
        4,
        "Two-Factor Exposure Analysis (Human/Livestock vs Farmland/Healthcare)",
        "scripts/eda/analyze_two_exposure_factors.py",
    )

    # Stage 5: Infrastructure risk spatial mapping
    run_stage(
        5,
        "Spatial Risk Mapping & Infrastructure Validation",
        "scripts/eda/rq1_spatial_risk_map.py",
    )

    # Stage 6: Random Forest predictive modeling
    run_stage(
        6,
        "Predictive Modeling: Random Forest Classification (GridSearchCV)",
        "scripts/eda/prediction_random_forest.py",
    )

    # Stage 7: Future flood forecasting
    if cli_args.skip_weather:
        print("\n[INFO] Skipping Stage 7 (live weather forecast) as requested via --skip-weather.")
    else:
        run_stage(
            7,
            "Future Flood Risk Prediction (RF Model x Live Open-Meteo Weather)",
            "scripts/eda/predict_future_floods.py",
        )

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 78)
    print(f" [PIPELINE COMPLETE] All stages executed successfully in {total_elapsed / 60:.1f} minutes!")
    print(f" Key datasets and figures are stored in: {OUTPUT_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()
