"""
Exposure Pillar Breakdown & Action-Targeted Flood Risk Modeling
===============================================================
This script extends the initial Sub-RQ 2 analysis by:
1. Comparing the additive risk index (Hazard + Exposure) with the standard
   humanitarian disaster risk formula: Risk = Hazard x Exposure.
2. Deconstructing the 4 exposure pillars (Population, Cattle, Cropland, Healthcare)
   for the top at-risk ADM3 payams in South Sudan.
3. Categorizing all 512 ADM3 payams into 3 targeted Action Tiers mapped
   directly to ZOA International's Anticipatory Action mandates:
   - Tier 1: Temporary Flood Defenses & Health Infrastructure Protection
   - Tier 2: Livestock Evacuation & Veterinary Protection
   - Tier 3: Pre-Flood Cash Transfers & Vulnerable Population Relief

Outputs:
- outputs/adm3_risk_and_action_priorities.csv
- outputs/top_risk_payams_pillar_breakdown.png
- outputs/risk_model_comparison_scatter.png
- outputs/zoa_intervention_priority_map.png
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ==============================================================================
# 1. Path Setup
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

INPUT_CSV = PROJECT_ROOT / "outputs" / "adm3_exposure_normalized.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RAW_DATA_DIR = PROJECT_ROOT / "raw_data" if (PROJECT_ROOT / "raw_data").exists() else PROJECT_ROOT / "data-JBG060-2026"
ADM3_GEOJSON = RAW_DATA_DIR / "Administrative boundaries" / "ssd_admin3.geojson"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 2. Load & Compute Multiplicative Risk Index
# ==============================================================================
def process_risk_and_action_tiers() -> pd.DataFrame:
    print("[INFO] Loading normalized ADM3 exposure data...")
    df = pd.read_csv(INPUT_CSV)

    # 1. Compute Standard Disaster Risk: Risk = Hazard x Exposure
    # (If Hazard is 0, Flood Risk is 0, correcting the additive flaw)
    df["multiplicative_risk"] = df["flood_hazard_2022_count_norm"] * df["combined_exposure_score"]
    df["additive_risk"] = df["flood_hazard_2022_count_norm"] + df["combined_exposure_score"]

    # 2. Flood Inundation Density (% of payam flooded)
    df["flood_density_per_sqkm"] = df["flood_hazard_2022_count"] / df["area_sqkm"]

    # 3. Define Action-Targeted Priority Tiers for ZOA International
    # Hazard threshold: top 20% of flooded payams
    hazard_thresh = df[df["flood_hazard_2022_count"] > 0]["flood_hazard_2022_count_norm"].quantile(0.60)
    
    def assign_action_tier(row):
        h = row["flood_hazard_2022_count_norm"]
        if h < 0.01:
            return "Minimal / No Flood Hazard"
        
        # Check specific exposure drivers
        pop_high = row["population_2024_norm"] >= 0.10 or row["health_facilities_count_norm"] >= 0.10
        cattle_high = row["cattle_sum_norm"] >= 0.10
        crop_high = row["cropland_score_norm"] >= 0.05

        if pop_high and h >= hazard_thresh:
            return "Tier 1: Temporary Defenses & Clinic Protection"
        elif cattle_high and h >= hazard_thresh:
            return "Tier 2: Livestock Evacuation & Vet Aid"
        elif crop_high and h >= hazard_thresh:
            return "Tier 3: Early Harvesting Trigger"
        elif h >= hazard_thresh:
            return "Tier 4: General Anticipatory Cash Transfers"
        else:
            return "Low-to-Moderate Flood Risk"

    df["zoa_action_tier"] = df.apply(assign_action_tier, axis=1)

    # Save enriched dataset
    out_csv = OUTPUT_DIR / "adm3_risk_and_action_priorities.csv"
    df.to_csv(out_csv, index=False)
    print(f"[SUCCESS] Saved risk and action priorities to:\n  -> {out_csv}")
    return df


# ==============================================================================
# 3. Visualizations
# ==============================================================================
def plot_top_payams_pillar_breakdown(df: pd.DataFrame):
    """Stacked & Grouped breakdown of the 4 exposure pillars for top 10 flood-risk payams."""
    print("[VISUALIZATION] Generating Top Payams Exposure Pillar Breakdown...")

    # Filter to top 10 true flood risk payams
    top10 = df.sort_values("multiplicative_risk", ascending=False).head(10).copy()
    top10 = top10.sort_values("multiplicative_risk", ascending=True)  # for horizontal plot

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1.8, 1]})

    # Left Panel: Stacked Exposure Pillars
    y_pos = np.arange(len(top10))
    payam_labels = [f"{row['adm3_name']} ({row['adm2_name']}, {row['adm1_name']})" for _, row in top10.iterrows()]

    p_pop = top10["population_2024_norm"]
    p_cat = top10["cattle_sum_norm"]
    p_crop = top10["cropland_score_norm"]
    p_health = top10["health_facilities_count_norm"]

    ax1.barh(y_pos, p_pop, label="Population Exposure", color="#2563EB", alpha=0.85)
    ax1.barh(y_pos, p_cat, left=p_pop, label="Cattle / Livestock Exposure", color="#D97706", alpha=0.85)
    ax1.barh(y_pos, p_crop, left=p_pop + p_cat, label="Cropland Exposure", color="#16A34A", alpha=0.85)
    ax1.barh(y_pos, p_health, left=p_pop + p_cat + p_crop, label="Healthcare Infrastructure", color="#DC2626", alpha=0.85)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(payam_labels, fontsize=9.5, fontweight="bold")
    ax1.set_xlabel("Cumulative Exposure Pillar Scores (Normalized)", fontsize=11, fontweight="bold")
    ax1.set_title("A. Exposure Composition of Top 10 Most Critical Payams\n(Ranked by True Multiplicative Flood Risk)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6, axis="x")
    ax1.legend(loc="lower right", fontsize=9, framealpha=0.95)

    # Right Panel: Flood Hazard Score for each
    bars_h = ax2.barh(y_pos, top10["flood_hazard_2022_count_norm"], color="#7C3AED", alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])  # share labels with ax1
    ax2.set_xlabel("Normalized Flood Hazard (2022)", fontsize=11, fontweight="bold")
    ax2.set_title("B. Inundation Hazard Severity", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6, axis="x")

    for bar in bars_h:
        w = bar.get_width()
        ax2.text(w + 0.02, bar.get_y() + bar.get_height()/2, f"{w:.2f}", va="center", fontsize=8.5, fontweight="bold")

    plt.suptitle("Sub-RQ 2 Deep-Dive: Deconstructing Vulnerability Across Top Inundated Payams", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "top_risk_payams_pillar_breakdown.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] {out_path.name}")


def plot_model_comparison_scatter(df: pd.DataFrame):
    """Compares the Additive vs Multiplicative Risk Models to show the Juba/Renk anomaly."""
    print("[VISUALIZATION] Generating Additive vs. Multiplicative Model Comparison...")

    fig, ax = plt.subplots(figsize=(10, 7))

    scatter = ax.scatter(
        df["combined_exposure_score"],
        df["flood_hazard_2022_count_norm"],
        c=df["multiplicative_risk"],
        cmap="YlOrRd",
        s=45,
        alpha=0.75,
        edgecolors="black",
        linewidths=0.3
    )

    # Highlight anomalous unflooded high-exposure zones vs true disaster hotspots
    anomalies = df[df["adm3_name"].isin(["Northern Bari", "Geger", "Juba Town"])]
    hotspots = df[df["adm3_name"].isin(["Rubkotne", "Budaang", "Biu", "Pajiek", "Manajang"])]

    for _, row in anomalies.iterrows():
        ax.annotate(
            f"{row['adm3_name']}\n(High Exposure, ZERO Flood)",
            (row["combined_exposure_score"], row["flood_hazard_2022_count_norm"]),
            xytext=(15, 20),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            color="navy",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8, edgecolor="blue"),
            arrowprops=dict(arrowstyle="->", color="blue", lw=1.2)
        )

    for _, row in hotspots.iterrows():
        ax.annotate(
            f"{row['adm3_name']} ({row['adm1_name']})\n★ True Disaster Hotspot",
            (row["combined_exposure_score"], row["flood_hazard_2022_count_norm"]),
            xytext=(-40, -35 if row["adm3_name"] == "Rubkotne" else 15),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            color="darkred",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEE2E2", alpha=0.9, edgecolor="red"),
            arrowprops=dict(arrowstyle="->", color="darkred", lw=1.2)
        )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("True Multiplicative Flood Risk (Hazard x Exposure)", fontsize=11, fontweight="bold")

    ax.set_xlabel("Combined Exposure Score (People, Cattle, Crops, Clinics)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Flood Hazard Score (2022 Satellite Detections)", fontsize=11, fontweight="bold")
    ax.set_title("Methodological Refinement: Correcting the 'Additive Risk Paradox'\n(Why Unflooded Urban & Crop Centers Must Not Be Ranked as Flood Disasters)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "risk_model_comparison_scatter.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] {out_path.name}")


def plot_zoa_action_map(df: pd.DataFrame):
    """Generates an ADM3 map classified into ZOA's Anticipatory Action Tiers."""
    if not ADM3_GEOJSON.exists():
        print(f"[WARN] Missing {ADM3_GEOJSON}. Skipping map generation.")
        return

    print("[VISUALIZATION] Generating ZOA Action Priority Map...")
    gdf = gpd.read_file(ADM3_GEOJSON)
    merged = gdf.merge(df[["adm3_pcode", "zoa_action_tier", "multiplicative_risk"]], on="adm3_pcode")

    fig, ax = plt.subplots(figsize=(12, 10))

    tier_colors = {
        "Tier 1: Temporary Defenses & Clinic Protection": "#DC2626",      # Bright Red
        "Tier 2: Livestock Evacuation & Vet Aid": "#EA580C",              # Orange
        "Tier 3: Early Harvesting Trigger": "#16A34A",                    # Green
        "Tier 4: General Anticipatory Cash Transfers": "#FACC15",         # Yellow
        "Low-to-Moderate Flood Risk": "#E2E8F0",                          # Light Gray
        "Minimal / No Flood Hazard": "#F8FAFC",                           # Off-white
    }

    import matplotlib.patches as mpatches

    legend_handles = []
    for tier, color in tier_colors.items():
        subset = merged[merged["zoa_action_tier"] == tier]
        if not subset.empty:
            subset.plot(ax=ax, color=color, edgecolor="#94A3B8", linewidth=0.3)
            legend_handles.append(mpatches.Patch(color=color, label=f"{tier} (n={len(subset)})"))

    ax.set_title("ZOA Anticipatory Action Operational Map\n(Prioritizing Interventions across South Sudan's 512 Payams)", fontsize=14, fontweight="bold")
    ax.axis("off")
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8.5, framealpha=0.95, title="ZOA Action Priorities", title_fontsize=9.5)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "zoa_intervention_priority_map.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] {out_path.name}")


# ==============================================================================
# 4. Main Execution
# ==============================================================================
def main():
    print("=" * 70)
    print("SUB-RQ 2 ENHANCEMENT: EXPOSURE DECONSTRUCTION & ACTION TIERS")
    print("=" * 70)

    df = process_risk_and_action_tiers()
    plot_top_payams_pillar_breakdown(df)
    plot_model_comparison_scatter(df)
    plot_zoa_action_map(df)

    print("\n[COMPLETE] All additional analyses and presentation figures generated successfully.\n")


if __name__ == "__main__":
    main()
