"""
Future Flood Risk Prediction — Live Weather Forecast + Trained RF Model
========================================================================
Combines two signals to predict near-future flood risk for every ADM3 payam:

  1. STRUCTURAL VULNERABILITY  (our trained Random Forest model)
     Which payams are inherently prone to flooding based on their
     geography, population density, livestock, cropland, and health
     infrastructure.

  2. WEATHER TRIGGER  (Open-Meteo API — free, no API key required)
     16-day daily precipitation forecasts fetched for the centroid of
     every payam.  Heavy rainfall on a structurally vulnerable payam
     dramatically raises the flood risk score.

Combined Risk Formula
---------------------
  daily_risk(payam, day) =
        α * structural_risk   +   β * rainfall_risk(day)
  where
    α = 0.5  (weight of structural vulnerability)
    β = 0.5  (weight of weather signal)
    rainfall_risk = normalised sigmoid of daily precipitation (mm)

Outputs (all in outputs/)
---------------------------------------------------------------------------
  future_flood_risk_daily.csv      — risk score per payam × per day (long)
  future_flood_risk_summary.csv    — peak / mean risk per payam over window
  future_flood_risk_heatmap.png    — heatmap: top payams × forecast days
  future_flood_risk_map.png        — geographic scatter: peak risk by location
  future_flood_risk_timeline.png   — daily risk timeline for top 10 payams
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

# ==============================================================================
# 0. Configuration
# ==============================================================================
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
OUTPUT_DIR   = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "random_forest_model.joblib"
DATA_CSV   = OUTPUT_DIR / "adm3_two_factor_exposure_analysis.csv"

# How many days ahead to forecast (Open-Meteo supports up to 16)
FORECAST_DAYS = 16

# Weight split between structural risk (α) and weather risk (β)
ALPHA = 0.5   # structural vulnerability weight
BETA  = 0.5   # weather / rainfall weight

# API rate-limiting: pause between requests (seconds)
API_PAUSE = 0.05   # 50 ms → ~512 payams in ~26 s

# Only score payams with structural risk above this threshold (speeds up API
# calls by skipping payams that are very unlikely to flood regardless of rain).
# Set to 0.0 to score all payams.
MIN_STRUCTURAL_RISK = 0.0


# ==============================================================================
# 1. Load Model & Data
# ==============================================================================
def load_model_and_data():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}.\n"
            "Run prediction_random_forest.py first to train and save the model."
        )
    print(f"[INFO] Loading trained Random Forest model from {MODEL_PATH.name} ...")
    model_bundle = joblib.load(MODEL_PATH)
    model        = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]
    print(f"       Features used: {feature_cols}")

    print(f"[INFO] Loading payam dataset from {DATA_CSV.name} ...")
    df = pd.read_csv(DATA_CSV)
    print(f"       {len(df)} payams loaded.")
    return model, feature_cols, df


# ==============================================================================
# 2. Compute Structural Vulnerability for Every Payam
# ==============================================================================
def compute_structural_risk(df: pd.DataFrame, model, feature_cols: list) -> pd.DataFrame:
    """
    Run every payam through the trained RF model to get the
    structural flood vulnerability probability (0–1).
    """
    print("[INFO] Computing structural flood vulnerability for all payams ...")
    score_df = df.dropna(subset=feature_cols + ["center_lat", "center_lon"]).copy()
    X = score_df[feature_cols]
    score_df["structural_risk"] = model.predict_proba(X)[:, 1]

    n_filtered = len(score_df[score_df["structural_risk"] >= MIN_STRUCTURAL_RISK])
    print(f"       {n_filtered} payams with structural risk ≥ {MIN_STRUCTURAL_RISK:.2f}")
    return score_df


# ==============================================================================
# 3. Fetch Weather Forecasts from Open-Meteo API
# ==============================================================================
OPEN_METEO_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude={lat}&longitude={lon}"
    "&daily=precipitation_sum,precipitation_probability_max"
    f"&forecast_days={FORECAST_DAYS}"
    "&timezone=auto"
)


def fetch_forecast(lat: float, lon: float, retries: int = 3) -> dict | None:
    """
    Fetch daily precipitation forecast for one location.
    Returns a dict {date_str: precipitation_mm} or None on failure.
    """
    url = OPEN_METEO_URL.format(lat=round(lat, 4), lon=round(lon, 4))
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read())
            daily = data["daily"]
            return dict(zip(daily["time"], daily["precipitation_sum"]))
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
            if attempt < retries - 1:
                time.sleep(1.0)
            else:
                print(f"  [WARN] Could not fetch forecast for ({lat:.4f}, {lon:.4f}): {exc}")
                return None


def fetch_all_forecasts(score_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch precipitation forecasts for every payam.
    Returns a DataFrame with columns: adm3_pcode, date, precipitation_mm
    """
    candidates = score_df[score_df["structural_risk"] >= MIN_STRUCTURAL_RISK].copy()
    print(f"\n[INFO] Fetching {FORECAST_DAYS}-day precipitation forecasts "
          f"for {len(candidates)} payams from Open-Meteo API ...")
    print("       (This takes ~30 seconds — free API, no key required)\n")

    rows = []
    for i, (_, row) in enumerate(candidates.iterrows(), 1):
        forecast = fetch_forecast(row["center_lat"], row["center_lon"])
        if forecast:
            for date, rain in forecast.items():
                rows.append({
                    "adm3_pcode": row["adm3_pcode"],
                    "adm3_name":  row["adm3_name"],
                    "date":       date,
                    "precipitation_mm": rain if rain is not None else 0.0,
                })
        if i % 50 == 0:
            print(f"       ... {i}/{len(candidates)} payams fetched")
        time.sleep(API_PAUSE)

    print(f"       Done — {len(rows)} (payam, day) records retrieved.")
    return pd.DataFrame(rows)


# ==============================================================================
# 4. Compute Combined Future Flood Risk
# ==============================================================================
def rainfall_to_risk(mm: float) -> float:
    """
    Convert daily rainfall (mm) into a 0–1 risk score via a sigmoid curve.

    Recalibrated for actual South Sudan rainfall levels in the Oct–Nov window
    (rainy season tail: typical forecasts 5–30 mm/day):
       0 mm  → 0.05   (dry)
       5 mm  → 0.25   (light rain)
      10 mm  → 0.45   (moderate — common in rainy season)
      15 mm  → 0.62   (significant rainfall)
      20 mm  → 0.75   (heavy — heightened risk)
      30 mm  → 0.90   (very heavy)
      40 mm  → 0.97   (extreme)
    """
    # Steeper sigmoid centred at 12 mm (realistic South Sudan trigger point)
    k        = 0.18
    midpoint = 12.0
    return float(np.clip(1.0 / (1.0 + np.exp(-k * (mm - midpoint))), 0.0, 1.0))


def compute_combined_risk(score_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge structural risk with daily weather risk and produce combined score.
    """
    print("[INFO] Computing combined future flood risk (structural + weather) ...")

    weather_df = weather_df.copy()
    weather_df["weather_risk"] = weather_df["precipitation_mm"].apply(rainfall_to_risk)

    # Merge with structural risk
    merged = weather_df.merge(
        score_df[["adm3_pcode", "adm1_name", "adm2_name", "structural_risk",
                  "center_lat", "center_lon",
                  "population_2024", "flood_hazard_2022_count"]],
        on="adm3_pcode",
        how="left",
    )

    merged["combined_risk"] = (
        ALPHA * merged["structural_risk"] + BETA * merged["weather_risk"]
    ).clip(0, 1)

    # Relative (percentile-based) tiers — always discriminating regardless of season:
    #   bottom 50 %  → Low
    #   50–75 %      → Moderate
    #   75–90 %      → High
    #   top 10 %     → Very High
    q50  = merged["combined_risk"].quantile(0.50)
    q75  = merged["combined_risk"].quantile(0.75)
    q90  = merged["combined_risk"].quantile(0.90)

    merged["risk_level"] = pd.cut(
        merged["combined_risk"],
        bins=[0, q50, q75, q90, 1.01],
        labels=["Low", "Moderate", "High", "Very High"],
        right=False,
        include_lowest=True,
    )

    return merged


# ==============================================================================
# 5. Summarise per Payam
# ==============================================================================
def summarise_by_payam(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily predictions to one row per payam."""
    summary = (
        daily_df.groupby(["adm3_pcode", "adm3_name", "adm1_name", "adm2_name",
                          "structural_risk", "center_lat", "center_lon",
                          "population_2024", "flood_hazard_2022_count"],
                         dropna=False)
        .agg(
            peak_combined_risk     =("combined_risk",     "max"),
            mean_combined_risk     =("combined_risk",     "mean"),
            peak_precipitation_mm  =("precipitation_mm",  "max"),
            total_precipitation_mm =("precipitation_mm",  "sum"),
            days_high_risk         =("combined_risk",
                                     lambda x: (x >= 0.70).sum()),
            peak_risk_date         =("combined_risk",
                                     lambda x: daily_df.loc[x.index, "date"].iloc[x.argmax()]),
        )
        .reset_index()
        .sort_values("peak_combined_risk", ascending=False)
    )

    # Percentile-based tiers (same logic as daily risk_level)
    q50 = summary["peak_combined_risk"].quantile(0.50)
    q75 = summary["peak_combined_risk"].quantile(0.75)
    q90 = summary["peak_combined_risk"].quantile(0.90)

    summary["overall_risk_tier"] = pd.cut(
        summary["peak_combined_risk"],
        bins=[0, q50, q75, q90, 1.01],
        labels=["Low", "Moderate", "High", "Very High"],
        right=False,
        include_lowest=True,
    )

    print(f"\n       Risk tier thresholds (relative to current forecast window):")
    print(f"         Low      < {q50:.3f}  (bottom 50 %)")
    print(f"         Moderate   {q50:.3f} – {q75:.3f}  (50–75 %)")
    print(f"         High       {q75:.3f} – {q90:.3f}  (75–90 %)")
    print(f"         Very High  > {q90:.3f}  (top 10 %)")
    return summary


# ==============================================================================
# 6. Visualisations
# ==============================================================================
def plot_heatmap(daily_df: pd.DataFrame, top_n: int = 25):
    """Heatmap: top N payams (rows) × forecast days (cols), colour = combined risk."""
    print("[VISUALIZATION] Generating forecast heatmap ...")

    # Pivot: top payams by peak risk
    pivot = daily_df.pivot_table(
        index="adm3_name", columns="date", values="combined_risk", aggfunc="mean"
    )
    top_payams = (
        daily_df.groupby("adm3_name")["combined_risk"].max()
        .nlargest(top_n).index
    )
    pivot = pivot.loc[pivot.index.isin(top_payams)].sort_values(
        by=pivot.columns.tolist(), ascending=False
    )

    fig, ax = plt.subplots(figsize=(14, max(7, len(pivot) * 0.38)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(
        [d[5:] for d in pivot.columns],  # strip year → "MM-DD"
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9, fontweight="bold")
    ax.set_xlabel("Forecast Date (MM-DD)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Future Flood Risk Forecast — Top {top_n} Payams\n"
        f"(Structural Vulnerability + Live Rainfall Forecast, next {FORECAST_DAYS} days)",
        fontsize=13, fontweight="bold",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Combined Flood Risk Score (0–1)", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out = OUTPUT_DIR / "future_flood_risk_heatmap.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[SAVED] {out.name}")


def plot_timeline(daily_df: pd.DataFrame, top_n: int = 10):
    """Line chart: daily combined risk over time for the top N payams."""
    print("[VISUALIZATION] Generating daily risk timeline ...")

    top_payams = (
        daily_df.groupby("adm3_name")["combined_risk"].max()
        .nlargest(top_n).index.tolist()
    )
    subset = daily_df[daily_df["adm3_name"].isin(top_payams)].copy()
    subset["date"] = pd.to_datetime(subset["date"])

    colors = plt.cm.tab10(np.linspace(0, 1, top_n))

    fig, ax = plt.subplots(figsize=(13, 6))
    for color, payam in zip(colors, top_payams):
        pdata = subset[subset["adm3_name"] == payam].sort_values("date")
        ax.plot(pdata["date"], pdata["combined_risk"], marker="o", ms=4,
                lw=2, label=payam, color=color)

    ax.axhline(0.70, color="red",    linestyle="--", lw=1.2, alpha=0.7, label="High-risk threshold (0.70)")
    ax.axhline(0.50, color="orange", linestyle=":",  lw=1.0, alpha=0.7, label="Moderate threshold (0.50)")

    ax.set_ylim(0, 1)
    ax.set_xlabel("Date", fontsize=11, fontweight="bold")
    ax.set_ylabel("Combined Flood Risk Score", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Daily Flood Risk Forecast — Top {top_n} Most At-Risk Payams\n"
        f"(next {FORECAST_DAYS} days, as of today)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9, ncol=2)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.autofmt_xdate()

    plt.tight_layout()
    out = OUTPUT_DIR / "future_flood_risk_timeline.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[SAVED] {out.name}")


def plot_map(summary_df: pd.DataFrame):
    """Scatter map: every payam coloured by peak combined risk."""
    print("[VISUALIZATION] Generating geographic risk map ...")

    fig, ax = plt.subplots(figsize=(10, 8))
    norm   = Normalize(vmin=0, vmax=1)
    cmap   = plt.cm.YlOrRd

    sc = ax.scatter(
        summary_df["center_lon"],
        summary_df["center_lat"],
        c=summary_df["peak_combined_risk"],
        cmap=cmap, norm=norm,
        s=summary_df["population_2024"].fillna(0).clip(0) / 5000 + 10,
        alpha=0.75,
        edgecolors="white", linewidths=0.3,
    )

    # Label the very high risk payams
    for _, row in summary_df[summary_df["peak_combined_risk"] >= 0.75].iterrows():
        ax.annotate(
            row["adm3_name"],
            xy=(row["center_lon"], row["center_lat"]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=6.5, color="#7F1D1D", fontweight="bold",
        )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Peak Combined Flood Risk (0–1)", fontsize=10, fontweight="bold")

    ax.set_xlabel("Longitude", fontsize=11, fontweight="bold")
    ax.set_ylabel("Latitude",  fontsize=11, fontweight="bold")
    ax.set_title(
        f"Forecasted Flood Risk Map — South Sudan\n"
        f"Peak risk over next {FORECAST_DAYS} days  "
        f"(dot size ∝ population)",
        fontsize=13, fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.4)

    plt.tight_layout()
    out = OUTPUT_DIR / "future_flood_risk_map.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[SAVED] {out.name}")


# ==============================================================================
# 7. Main
# ==============================================================================
def main():
    print("=" * 70)
    print("FUTURE FLOOD RISK PREDICTION")
    print("Trained RF Model  ×  Live 16-Day Rainfall Forecast (Open-Meteo)")
    print("=" * 70)

    # ── Load ────────────────────────────────────────────────────────────────
    model, feature_cols, df = load_model_and_data()

    # ── Structural risk from trained RF model ────────────────────────────────
    score_df = compute_structural_risk(df, model, feature_cols)

    # ── Live weather forecasts from API ──────────────────────────────────────
    weather_df = fetch_all_forecasts(score_df)

    if weather_df.empty:
        print("[ERROR] No weather data retrieved. Check your internet connection.")
        return

    # ── Combine signals ──────────────────────────────────────────────────────
    daily_df = compute_combined_risk(score_df, weather_df)

    # ── Save daily long-format results ───────────────────────────────────────
    daily_out = OUTPUT_DIR / "future_flood_risk_daily.csv"
    daily_df.sort_values(["combined_risk", "date"], ascending=[False, True]).to_csv(
        daily_out, index=False
    )
    print(f"\n[SAVED] Daily risk data → {daily_out.name}")

    # ── Summarise per payam ──────────────────────────────────────────────────
    summary_df = summarise_by_payam(daily_df)
    summary_out = OUTPUT_DIR / "future_flood_risk_summary.csv"
    summary_df.to_csv(summary_out, index=False)
    print(f"[SAVED] Summary per payam → {summary_out.name}")

    # ── Visualisations ───────────────────────────────────────────────────────
    plot_heatmap(daily_df)
    plot_timeline(daily_df)
    plot_map(summary_df)

    # ── Console summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TOP 15 PAYAMS — HIGHEST FORECASTED FLOOD RISK (next 16 days)")
    print("=" * 70)
    display_cols = [
        "adm3_name", "adm2_name", "adm1_name",
        "structural_risk", "peak_precipitation_mm",
        "peak_combined_risk", "days_high_risk",
        "peak_risk_date", "overall_risk_tier",
    ]
    available = [c for c in display_cols if c in summary_df.columns]
    print(summary_df.head(15)[available].to_string(index=False))

    print(f"\n[COMPLETE] Future flood risk prediction finished.\n"
          f"  All outputs saved to: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
