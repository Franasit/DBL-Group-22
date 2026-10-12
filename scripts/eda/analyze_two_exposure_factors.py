"""
Two-Factor Exposure Analysis (Sub-RQ 2 Deep Dive)
=================================================
Analyzes two distinct exposure composites across all 512 ADM3 payams:
1. Factor 1: Exposure: Population + Livestock (Human & Pastoral Capital)
2. Factor 2: Exposure: Farmland / Livelihood + Infrastructure (Agronomic Production & Critical Health Services)

Computes both composite exposure and true flood risk (Hazard x Factor Exposure),
identifying geographic divergence, top hotspot payams, and strategic ZOA operational targets.

Outputs:
- outputs/adm3_two_factor_exposure_analysis.csv
- outputs/two_factor_comparison_scatter.png
- outputs/top_payams_two_factors_breakdown.png
- outputs/two_factor_dual_risk_maps.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ==============================================================================
# 1. Setup & Paths
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

INPUT_CSV = PROJECT_ROOT / "outputs" / "adm3_exposure_normalized.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RAW_DATA_DIR = PROJECT_ROOT / "raw_data" if (PROJECT_ROOT / "raw_data").exists() else PROJECT_ROOT / "data-JBG060-2026"
ADM3_GEOJSON = RAW_DATA_DIR / "Administrative boundaries" / "ssd_admin3.geojson"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 2. Factor Formulation & Calculation
# ==============================================================================
def compute_two_factors() -> pd.DataFrame:
    print("[INFO] Loading normalized ADM3 exposure dataset...")
    df = pd.read_csv(INPUT_CSV)

    # ---------------------------------------------------------
    # Factor 1: Population + Livestock (Human & Pastoralist Capital)
    # ---------------------------------------------------------
    # Average of normalized population and normalized cattle sum
    df["factor1_pop_livestock_exposure"] = (df["population_2024_norm"] + df["cattle_sum_norm"]) / 2.0
    # Multiplicative Risk = Hazard x Factor Exposure
    df["factor1_pop_livestock_risk"] = df["flood_hazard_2022_count_norm"] * df["factor1_pop_livestock_exposure"]

    # ---------------------------------------------------------
    # Factor 2: Farmland + Infrastructure (Cropland & Healthcare Services)
    # ---------------------------------------------------------
    # Average of normalized cropland score and normalized health facilities count
    df["factor2_farm_infra_exposure"] = (df["cropland_score_norm"] + df["health_facilities_count_norm"]) / 2.0
    # Multiplicative Risk = Hazard x Factor Exposure
    df["factor2_farm_infra_risk"] = df["flood_hazard_2022_count_norm"] * df["factor2_farm_infra_exposure"]

    # Relative dominance: Which factor drives the vulnerability in this payam?
    def determine_dominant_profile(row):
        r1 = row["factor1_pop_livestock_risk"]
        r2 = row["factor2_farm_infra_risk"]
        if r1 < 0.005 and r2 < 0.005:
            return "Low / Unaffected"
        elif r1 >= 1.5 * r2:
            return "Dominant: Pastoral & Human Crisis (Pop + Cattle)"
        elif r2 >= 1.5 * r1:
            return "Dominant: Agricultural & Service Crisis (Crops + Clinics)"
        else:
            return "Dual Crisis: High Pastoral & Service Exposure"

    df["dominant_crisis_profile"] = df.apply(determine_dominant_profile, axis=1)

    # Save to CSV
    out_csv = OUTPUT_DIR / "adm3_two_factor_exposure_analysis.csv"
    df.to_csv(out_csv, index=False)
    print(f"[SUCCESS] Saved two-factor dataset to:\n  -> {out_csv}")
    return df


# ==============================================================================
# 3. Visualizations
# ==============================================================================
def plot_factor_comparison_scatter(df: pd.DataFrame):
    """Scatter plot comparing Factor 1 (Pop + Cattle) vs Factor 2 (Crops + Clinics) flood risk."""
    print("[VISUALIZATION] Generating Two-Factor Scatter Comparison...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot
    scatter = ax.scatter(
        df["factor1_pop_livestock_risk"],
        df["factor2_farm_infra_risk"],
        c=df["flood_hazard_2022_count_norm"],
        cmap="YlOrRd",
        s=55,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.4
    )

    # Annotate top key payams representing different crisis archetypes
    top_labels = [
        ("Rubkotne", (15, 10)),
        ("Budaang", (-50, 15)),
        ("Biu", (15, -15)),
        ("Pajiek", (15, 10)),
        ("Manajang", (-70, 15)),
        ("Nhialdiu", (15, 5)),
        ("Narus", (15, 10)),
        ("Abyei Region", (15, -10)),
        ("Kuac North", (15, 10)),
        ("Tiar-aleit", (15, -10))
    ]

    for name, offset in top_labels:
        match = df[df["adm3_name"] == name]
        if not match.empty:
            row = match.iloc[0]
            ax.annotate(
                f"{name} ({row['adm2_name']})",
                (row["factor1_pop_livestock_risk"], row["factor2_farm_infra_risk"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=8.5,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#64748B"),
                arrowprops=dict(arrowstyle="->", color="#334155", lw=0.8)
            )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Normalized Flood Hazard Intensity (2022)", fontsize=11, fontweight="bold")

    # Reference 1:1 diagonal line
    max_val = max(df["factor1_pop_livestock_risk"].max(), df["factor2_farm_infra_risk"].max()) * 1.05
    ax.plot([0, max_val], [0, max_val], color="gray", linestyle="--", alpha=0.5, label="Equal Risk Balance Line (1:1)")

    ax.set_xlabel("Factor 1 Flood Risk: Population + Livestock (Pastoral Human Capital)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Factor 2 Flood Risk: Farmland + Infrastructure (Crops & Healthcare)", fontsize=11, fontweight="bold")
    ax.set_title("Cross-Factor Exposure Analysis: Pastoral Crisis vs. Agricultural/Service Crisis", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "two_factor_comparison_scatter.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] {out_path.name}")


def plot_top_payams_factor_breakdown(df: pd.DataFrame):
    """Grouped bar chart comparing Factor 1 vs Factor 2 risk for the top 10 overall risk payams."""
    print("[VISUALIZATION] Generating Top Payams Factor Breakdown Chart...")

    df["overall_risk"] = df["factor1_pop_livestock_risk"] + df["factor2_farm_infra_risk"]
    top10 = df.sort_values("overall_risk", ascending=False).head(10).copy()
    top10 = top10.sort_values("overall_risk", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    y = np.arange(len(top10))
    height = 0.38

    labels = [f"{row['adm3_name']} ({row['adm2_name']}, {row['adm1_name']})" for _, row in top10.iterrows()]

    b1 = ax.barh(y - height/2, top10["factor1_pop_livestock_risk"], height, label="Factor 1: Population + Livestock Risk", color="#2563EB", alpha=0.85)
    b2 = ax.barh(y + height/2, top10["factor2_farm_infra_risk"], height, label="Factor 2: Farmland + Infrastructure Risk", color="#EA580C", alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_xlabel("Normalized Flood Risk Score (Hazard x Factor Exposure)", fontsize=11, fontweight="bold")
    ax.set_title("Comparative Impact Breakdown: Top 10 Most Affected ADM3 Payams\n(Population & Cattle Threat vs. Cropland & Clinic Threat)", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6, axis="x")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)

    # Annotate values on bars
    for bar in b1:
        w = bar.get_width()
        ax.text(w + 0.003, bar.get_y() + bar.get_height()/2, f"{w:.3f}", va="center", fontsize=8, color="#1E3A8A", fontweight="bold")
    for bar in b2:
        w = bar.get_width()
        ax.text(w + 0.003, bar.get_y() + bar.get_height()/2, f"{w:.3f}", va="center", fontsize=8, color="#9A3412", fontweight="bold")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "top_payams_two_factors_breakdown.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] {out_path.name}")


def plot_dual_factor_maps(df: pd.DataFrame):
    """Side-by-side geographic maps comparing Factor 1 Flood Risk vs Factor 2 Flood Risk."""
    if not ADM3_GEOJSON.exists():
        print(f"[WARN] Missing {ADM3_GEOJSON}. Skipping map generation.")
        return

    print("[VISUALIZATION] Generating Dual-Factor National Risk Maps...")
    gdf = gpd.read_file(ADM3_GEOJSON)
    merged = gdf.merge(df[["adm3_pcode", "factor1_pop_livestock_risk", "factor2_farm_infra_risk"]], on="adm3_pcode")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

    cmap = "YlOrRd"

    # Map 1: Factor 1 Risk
    norm1 = mcolors.Normalize(vmin=0, vmax=max(merged["factor1_pop_livestock_risk"].max(), 0.05))
    merged.plot(column="factor1_pop_livestock_risk", cmap=cmap, norm=norm1, edgecolor="#94A3B8", linewidth=0.25, ax=ax1)
    ax1.set_title("A. Factor 1 Flood Risk: Population + Livestock\n(Concentrated in Unity & Jonglei Sudd Wetlands)", fontsize=12, fontweight="bold")
    ax1.axis("off")
    sm1 = plt.cm.ScalarMappable(cmap=cmap, norm=norm1)
    sm1.set_array([])
    cbar1 = fig.colorbar(sm1, ax=ax1, orientation="horizontal", shrink=0.7, pad=0.03)
    cbar1.set_label("Factor 1 Risk Score (Hazard x Pop/Livestock Exposure)", fontsize=10, fontweight="bold")

    # Map 2: Factor 2 Risk
    norm2 = mcolors.Normalize(vmin=0, vmax=max(merged["factor2_farm_infra_risk"].max(), 0.05))
    merged.plot(column="factor2_farm_infra_risk", cmap=cmap, norm=norm2, edgecolor="#94A3B8", linewidth=0.25, ax=ax2)
    ax2.set_title("B. Factor 2 Flood Risk: Farmland + Infrastructure\n(Extending into Warrap, NBeG & Upper Nile Fringes)", fontsize=12, fontweight="bold")
    ax2.axis("off")
    sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm2)
    sm2.set_array([])
    cbar2 = fig.colorbar(sm2, ax=ax2, orientation="horizontal", shrink=0.7, pad=0.03)
    cbar2.set_label("Factor 2 Risk Score (Hazard x Farmland/Clinic Exposure)", fontsize=10, fontweight="bold")

    plt.suptitle("Geographic Disaggregation of Flood Impacts at the ADM3 (Payam) Level", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "two_factor_dual_risk_maps.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] {out_path.name}")


# ==============================================================================
# 4. Main Execution
# ==============================================================================
def main():
    print("=" * 70)
    print("TWO-FACTOR EXPOSURE ANALYSIS: FACTOR 1 vs. FACTOR 2")
    print("=" * 70)

    df = compute_two_factors()
    plot_factor_comparison_scatter(df)
    plot_top_payams_factor_breakdown(df)
    plot_dual_factor_maps(df)

    print("\n[COMPLETE] Two-factor exposure analysis finished successfully.\n")


if __name__ == "__main__":
    main()
