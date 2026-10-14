"""
South Sudan Flood Risk Maps
===========================
Produces two publication-quality maps using real ADM3 polygon boundaries:

  Map 1 — Future Flood Risk (next 16 days)
           Choropleth coloured by peak_combined_risk from the RF model
           + live weather forecast.  ADM1 state borders overlaid.

  Map 2 — Historical Flood Intensity (2022)
           Choropleth coloured by flood_hazard_2022_count for reference /
           comparison with the future forecast.

Outputs:
  outputs/map_future_flood_risk.png
  outputs/map_historical_flood_2022.png
  outputs/map_flood_risk_side_by_side.png
"""

from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ==============================================================================
# Paths
# ==============================================================================
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR     = PROJECT_ROOT / "data-JBG060-2026" / "Administrative boundaries"
OUTPUT_DIR   = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ADM3_GEOJSON = DATA_DIR / "ssd_admin3.geojson"
ADM1_GEOJSON = DATA_DIR / "ssd_admin1.geojson"
ADM0_GEOJSON = DATA_DIR / "ssd_admin0.geojson"

RISK_CSV     = OUTPUT_DIR / "future_flood_risk_summary.csv"

TIER_COLORS  = {
    "Low":       "#93C5FD",   # light blue
    "Moderate":  "#FCD34D",   # yellow
    "High":      "#F97316",   # orange
    "Very High": "#DC2626",   # red
}

# ==============================================================================
# 1. Load Data
# ==============================================================================
def load_data():
    print("[INFO] Loading geographic boundaries ...")
    gdf3 = gpd.read_file(ADM3_GEOJSON)   # payam polygons
    gdf1 = gpd.read_file(ADM1_GEOJSON)   # state borders
    gdf0 = gpd.read_file(ADM0_GEOJSON)   # country border

    print("[INFO] Loading future flood risk scores ...")
    risk_df = pd.read_csv(RISK_CSV)

    print(f"       {len(gdf3)} payam polygons, {len(risk_df)} risk records")
    return gdf3, gdf1, gdf0, risk_df


# ==============================================================================
# 2. Merge geometries with risk data
# ==============================================================================
def merge_risk(gdf3: gpd.GeoDataFrame, risk_df: pd.DataFrame) -> gpd.GeoDataFrame:
    merged = gdf3.merge(
        risk_df[["adm3_pcode", "peak_combined_risk", "mean_combined_risk",
                 "overall_risk_tier", "days_high_risk", "peak_risk_date",
                 "peak_precipitation_mm", "total_precipitation_mm",
                 "structural_risk", "flood_hazard_2022_count"]],
        on="adm3_pcode",
        how="left",
    )
    n_missing = merged["peak_combined_risk"].isna().sum()
    if n_missing:
        print(f"  [WARN] {n_missing} payams had no risk score (API miss) — shown in grey")
    return merged


# ==============================================================================
# 3. Map 1 — Future Flood Risk Choropleth
# ==============================================================================
def plot_future_risk(merged: gpd.GeoDataFrame, gdf1: gpd.GeoDataFrame,
                     gdf0: gpd.GeoDataFrame, out_path: Path):
    print("[VISUALIZATION] Rendering future flood risk map ...")

    fig, ax = plt.subplots(figsize=(14, 11))
    fig.patch.set_facecolor("#F0F4F8")
    ax.set_facecolor("#C8E6F5")   # light blue = water / background

    # ── Country fill (base) ──────────────────────────────────────────────────
    gdf0.plot(ax=ax, color="#EEE8D5", edgecolor="none", zorder=1)

    # ── Payam polygons: grey for no data ────────────────────────────────────
    no_data = merged[merged["peak_combined_risk"].isna()]
    no_data.plot(ax=ax, color="#D1D5DB", edgecolor="#9CA3AF",
                 linewidth=0.2, zorder=2)

    # ── Payam polygons: coloured by risk score (continuous colormap) ─────────
    has_data = merged[merged["peak_combined_risk"].notna()]
    cmap     = mpl.cm.RdYlGn_r          # green (low) → red (high)
    norm     = mcolors.Normalize(vmin=0, vmax=1)

    has_data.plot(
        column="peak_combined_risk",
        ax=ax,
        cmap=cmap,
        norm=norm,
        edgecolor="#6B7280",
        linewidth=0.15,
        zorder=3,
    )

    # ── State (ADM1) borders — bold overlay ──────────────────────────────────
    gdf1.plot(ax=ax, color="none", edgecolor="#1F2937",
              linewidth=1.4, zorder=4)

    # ── Country border ───────────────────────────────────────────────────────
    gdf0.plot(ax=ax, color="none", edgecolor="#111827",
              linewidth=2.0, zorder=5)

    # ── Label state names ────────────────────────────────────────────────────
    for _, row in gdf1.iterrows():
        cx = row.geometry.centroid.x
        cy = row.geometry.centroid.y
        ax.annotate(
            row["adm1_name"],
            xy=(cx, cy),
            ha="center", va="center",
            fontsize=7.5, fontweight="bold",
            color="#1F2937",
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.55, ec="none"),
        )

    # ── Label top 10 highest-risk payams ─────────────────────────────────────
    top10 = has_data.nlargest(10, "peak_combined_risk")
    for _, row in top10.iterrows():
        cx = row.geometry.centroid.x
        cy = row.geometry.centroid.y
        ax.annotate(
            row["adm3_name"],
            xy=(cx, cy),
            xytext=(5, 5), textcoords="offset points",
            fontsize=6.5, color="#7F1D1D", fontweight="bold",
            zorder=7,
            arrowprops=dict(arrowstyle="-", color="#7F1D1D", lw=0.8),
        )

    # ── Colourbar ────────────────────────────────────────────────────────────
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01, shrink=0.75)
    cbar.set_label("Peak Combined Flood Risk Score  (0 = none, 1 = extreme)",
                   fontsize=10, fontweight="bold")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=9)

    # ── Tier legend (discrete) ───────────────────────────────────────────────
    tier_patches = [
        mpatches.Patch(color="#DC2626", label="Very High  (top 10 %)"),
        mpatches.Patch(color="#F97316", label="High          (75–90 %)"),
        mpatches.Patch(color="#FCD34D", label="Moderate  (50–75 %)"),
        mpatches.Patch(color="#4ADE80", label="Low            (bottom 50 %)"),
        mpatches.Patch(color="#D1D5DB", label="No forecast data"),
    ]
    legend = ax.legend(handles=tier_patches, title="Risk Tier", title_fontsize=9,
              fontsize=8.5, loc="lower left", framealpha=0.9,
              edgecolor="#9CA3AF")
    legend.set_zorder(8)

    # ── Title & labels ───────────────────────────────────────────────────────
    ax.set_title(
        "South Sudan — Forecasted Flood Risk by Payam (ADM3)\n"
        "Next 16 Days  |  RF Structural Vulnerability + Live Rainfall Forecast (Open-Meteo)",
        fontsize=14, fontweight="bold", pad=14,
    )
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude",  fontsize=10)
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path.name}")


# ==============================================================================
# 4. Map 2 — Historical Flood Intensity 2022
# ==============================================================================
def plot_historical_flood(merged: gpd.GeoDataFrame, gdf1: gpd.GeoDataFrame,
                           gdf0: gpd.GeoDataFrame, out_path: Path):
    print("[VISUALIZATION] Rendering historical 2022 flood map ...")

    fig, ax = plt.subplots(figsize=(14, 11))
    fig.patch.set_facecolor("#F0F4F8")
    ax.set_facecolor("#C8E6F5")

    gdf0.plot(ax=ax, color="#EEE8D5", edgecolor="none", zorder=1)

    flood_col = "flood_hazard_2022_count"
    merged["flood_log"] = np.log1p(merged[flood_col].fillna(0))

    cmap = mpl.cm.Blues
    norm = mcolors.Normalize(
        vmin=0, vmax=merged["flood_log"].quantile(0.98)
    )

    merged.plot(
        column="flood_log",
        ax=ax,
        cmap=cmap,
        norm=norm,
        edgecolor="#6B7280",
        linewidth=0.15,
        zorder=2,
        missing_kwds={"color": "#D1D5DB"},
    )

    gdf1.plot(ax=ax, color="none", edgecolor="#1F2937",
              linewidth=1.4, zorder=3)
    gdf0.plot(ax=ax, color="none", edgecolor="#111827",
              linewidth=2.0, zorder=4)

    for _, row in gdf1.iterrows():
        ax.annotate(
            row["adm1_name"],
            xy=(row.geometry.centroid.x, row.geometry.centroid.y),
            ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="#1F2937",
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.55, ec="none"),
        )

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01, shrink=0.75)
    cbar.set_label("Flood Hazard Count 2022  (log scale)", fontsize=10, fontweight="bold")
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(
        "South Sudan — Historical Flood Intensity by Payam (ADM3)\n"
        "2022 Flood Hazard Count  |  Source: Historical Data",
        fontsize=14, fontweight="bold", pad=14,
    )
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude",  fontsize=10)
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path.name}")


# ==============================================================================
# 5. Map 3 — Side-by-Side Comparison
# ==============================================================================
def plot_side_by_side(merged: gpd.GeoDataFrame, gdf1: gpd.GeoDataFrame,
                      gdf0: gpd.GeoDataFrame, out_path: Path):
    print("[VISUALIZATION] Rendering side-by-side comparison map ...")

    fig, axes = plt.subplots(1, 2, figsize=(22, 11))
    fig.patch.set_facecolor("#F0F4F8")

    panels = [
        ("peak_combined_risk", mpl.cm.RdYlGn_r,
         mcolors.Normalize(0, 1),
         "Future Risk Score (next 16 days)"),
        ("flood_log",          mpl.cm.Blues,
         mcolors.Normalize(0, merged["flood_log"].quantile(0.98)),
         "Historical Flood Intensity 2022  (log scale)"),
    ]

    for ax, (col, cmap, norm, subtitle) in zip(axes, panels):
        ax.set_facecolor("#C8E6F5")
        gdf0.plot(ax=ax, color="#EEE8D5", edgecolor="none", zorder=1)
        merged.plot(column=col, ax=ax, cmap=cmap, norm=norm,
                    edgecolor="#6B7280", linewidth=0.15, zorder=2,
                    missing_kwds={"color": "#D1D5DB"})
        gdf1.plot(ax=ax, color="none", edgecolor="#1F2937",
                  linewidth=1.2, zorder=3)
        gdf0.plot(ax=ax, color="none", edgecolor="#111827",
                  linewidth=1.8, zorder=4)

        for _, row in gdf1.iterrows():
            ax.annotate(
                row["adm1_name"],
                xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                ha="center", va="center", fontsize=7, fontweight="bold",
                color="#1F2937", zorder=5,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", alpha=0.5, ec="none"),
            )

        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01, shrink=0.72)
        cbar.ax.tick_params(labelsize=8)

        ax.set_title(subtitle, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Longitude", fontsize=9)
        ax.set_ylabel("Latitude",  fontsize=9)
        ax.tick_params(labelsize=8)

    fig.suptitle(
        "South Sudan — Flood Risk: Future Forecast vs. Historical 2022",
        fontsize=16, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path.name}")


# ==============================================================================
# Main
# ==============================================================================
def main():
    print("=" * 65)
    print("SOUTH SUDAN FLOOD RISK — MAP GENERATION")
    print("=" * 65)

    gdf3, gdf1, gdf0, risk_df = load_data()
    merged = merge_risk(gdf3, risk_df)
    merged["flood_log"] = np.log1p(merged["flood_hazard_2022_count"].fillna(0))

    plot_future_risk(
        merged, gdf1, gdf0,
        OUTPUT_DIR / "map_future_flood_risk.png"
    )
    plot_historical_flood(
        merged, gdf1, gdf0,
        OUTPUT_DIR / "map_historical_flood_2022.png"
    )
    plot_side_by_side(
        merged, gdf1, gdf0,
        OUTPUT_DIR / "map_flood_risk_side_by_side.png"
    )

    print("\n[COMPLETE] All maps saved to outputs/")


if __name__ == "__main__":
    main()
