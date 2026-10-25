"""JBG060 — Script 26: ADM3 flood-hazard poster maps with Top-10 callouts.

Inputs: Script 25 FINAL v2 synthesis and official ADM2/ADM3 GeoJSON boundaries.
Outputs: original maps, the Top-10 hazard poster figure, three additional
Top-10 poster figures (extent, change, robust hotspots), and QC tables.

No hazard classification or score is recalculated. This script visualizes the
existing hazard-screening outputs; it does NOT rank humanitarian priority.
"""
from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import TwoSlopeNorm, BoundaryNorm, ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, Rectangle

PROJECT_ROOT = Path("/Users/shibai/Documents/GitHub/DBL-Group-22")
INPUT_CSV = (PROJECT_ROOT / "processed_data/cross_dataset_analysis/03_flood_composition/"
             "25_adm3_flood_hazard_synthesis/tables/25_adm3_flood_hazard_synthesis.csv")
ADM3_GEOJSON = PROJECT_ROOT / "raw_data/administrative_boundaries/ssd_admin3.geojson"
ADM2_GEOJSON = PROJECT_ROOT / "raw_data/administrative_boundaries/ssd_admin2.geojson"
OUT = PROJECT_ROOT / "processed_data/cross_dataset_analysis/06_poster_results/26_map_adm3_final_flood_hazard"
FIG = OUT / "figures"
TAB = OUT / "tables"
QC = OUT / "qc"
DPI = 300
TOP_N = 10

CLASS_COLORS = {
    "Stable / lower local hazard": "#d9d9d9",         # light gray
    "Moderate increasing hazard": "#ffd92f",          # yellow
    "Emerging / worsening local hazard": "#ff9f1c",   # orange
    "Persistent / high recent hazard": "#e85d04",     # orange-red
    "High recent and worsening hazard": "#d62828",    # red
    "Robust local amplification": "#7b2cbf",          # purple
    "Robust high local hazard": "#3a0ca3",            # deep violet
}


def norm(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value).lower())).strip()


def choose(df, options, context, required=True):
    cols = {str(c).lower(): c for c in df.columns}
    for candidate in options:
        if candidate.lower() in cols:
            return cols[candidate.lower()]
    if required:
        raise ValueError(f"Missing {context}; expected one of {options}. Available columns: {list(df.columns)}")
    return None


def require(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required input missing: {path}")


def make_name_keys(df):
    a2 = choose(df, ["ADM2_NAME_STD", "ADM2_NAME", "adm2_name", "NAME_2"], "parent ADM2 name")
    a3 = choose(df, ["ADM3_NAME_STD", "ADM3_NAME", "adm3_name", "NAME_3"], "ADM3 name")
    return df[a2].map(norm) + "|" + df[a3].map(norm), "ADM2+ADM3 names"


def make_key(df, adm3=True):
    id_opts = (["ADM3_ID", "adm3_pcode", "ADM3_PCODE"] if adm3 else
               ["ADM2_ID", "adm2_pcode", "ADM2_PCODE"])
    code = choose(df, id_opts, "ADM ID", required=False)
    if code is not None:
        pcodes = df[code].astype("string").str.strip().str.upper()
        if pcodes.notna().all() and pcodes.ne("").all() and pcodes.is_unique:
            return pcodes.astype(str), "PCODE"
    return make_name_keys(df)


def prepare():
    for path in [INPUT_CSV, ADM3_GEOJSON, ADM2_GEOJSON]:
        require(path)
    flood = pd.read_csv(INPUT_CSV)
    bounds3 = gpd.read_file(ADM3_GEOJSON)
    bounds2 = gpd.read_file(ADM2_GEOJSON)
    if bounds3.crs is None or bounds2.crs is None:
        raise ValueError("Boundary CRS is missing; cannot map reliably.")
    if len(flood) != 512 or len(bounds3) != 512 or len(bounds2) != 79:
        raise ValueError(f"Unexpected counts: synthesis={len(flood)}, ADM3 boundary={len(bounds3)}, ADM2 boundary={len(bounds2)}. Expected 512/512/79.")

    needed = [
        "final_hazard_screening_class",
        "recent_combined_mean",
        "combined_change_recent_minus_early",
        "robust_local_amplification_flag",
        "final_hazard_score_0_100",
    ]
    missing = [x for x in needed if x not in flood.columns]
    if missing:
        raise ValueError(f"Script 25 FINAL v2 fields missing: {missing}. Available: {list(flood.columns)}")

    flood = flood.copy()
    flood["_key"], source_type = make_key(flood)
    boundary_code = choose(bounds3, ["ADM3_ID", "adm3_pcode", "ADM3_PCODE"], "ADM3 boundary PCODE", required=False)
    flood_code = choose(flood, ["ADM3_ID", "adm3_pcode", "ADM3_PCODE"], "Script 25 ADM3 PCODE", required=False)
    using_pcode = source_type == "PCODE" and boundary_code is not None and flood_code is not None
    bounds3 = bounds3.copy()
    if using_pcode:
        bounds3["_key"] = bounds3[boundary_code].astype("string").str.strip().str.upper().astype(str)
        method = "ADM3 PCODE"
    else:
        flood["_key"], _ = make_name_keys(flood)
        bounds3["_key"], _ = make_name_keys(bounds3)
        method = "ADM2+ADM3 names"

    mapped = bounds3[["_key", "geometry"]].merge(flood, on="_key", how="left", validate="one_to_one", indicator=True)
    mapped = gpd.GeoDataFrame(mapped, geometry="geometry", crs=bounds3.crs)
    unmatched = mapped[mapped["_merge"] != "both"]
    unmatched[["_key", "_merge"]].to_csv(TAB / "26_unmatched_boundary_keys.csv", index=False)
    if len(unmatched):
        raise ValueError(f"{len(unmatched)} unmatched ADM3 polygons using {method}. See tables/26_unmatched_boundary_keys.csv.")
    mapped.drop(columns="_merge", inplace=True)

    unknown = set(mapped["final_hazard_screening_class"].dropna().unique()) - set(CLASS_COLORS)
    if unknown:
        raise ValueError(f"Unexpected Script 25 hazard classes: {sorted(unknown)}")

    for name in ["recent_combined_mean", "combined_change_recent_minus_early", "final_hazard_score_0_100"]:
        mapped[name] = pd.to_numeric(mapped[name], errors="coerce")

    flag = mapped["robust_local_amplification_flag"].astype("string").str.strip().str.lower()
    if not flag.isin(["true", "false", "1", "0"]).all():
        raise ValueError("Unrecognized robust_local_amplification_flag values")
    mapped["robust_flag"] = flag.isin(["true", "1"])
    return mapped, bounds2.to_crs(mapped.crs), method


def base_plot(ax, a2):
    a2.boundary.plot(ax=ax, color="#555555", linewidth=.65, zorder=5)
    ax.set_axis_off()
    ax.set_aspect("equal")


def export(fig, name):
    fig.savefig(FIG / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", FIG / name)


def build_top10_table(m):
    names = choose(m, ["ADM3_NAME_STD", "ADM3_NAME"], "ADM3 name")
    parents = choose(m, ["ADM2_NAME_STD", "ADM2_NAME"], "ADM2 name")
    top = m.dropna(subset=["final_hazard_score_0_100"]).nlargest(TOP_N, "final_hazard_score_0_100").copy()
    top = top.reset_index(drop=True)
    top["rank_no"] = range(1, len(top) + 1)
    top["label_adm3"] = top[names].astype(str)
    top["label_adm2"] = top[parents].astype(str)
    top["label_score"] = top["final_hazard_score_0_100"].round(2)
    reps = top.geometry.representative_point()
    top["label_x"] = reps.x
    top["label_y"] = reps.y
    cols = ["rank_no", "label_adm3", "label_adm2", "label_score", "final_hazard_screening_class", "_key"]
    top[cols].to_csv(TAB / "26_top10_adm3_map_labels.csv", index=False)
    return top


def overlay_priority_outlines(ax, m):
    """Emphasize the two most important robust classes for poster readability."""
    robust_high = m[m["final_hazard_screening_class"] == "Robust high local hazard"]
    robust_local = m[m["final_hazard_screening_class"] == "Robust local amplification"]
    if len(robust_high):
        robust_high.boundary.plot(ax=ax, color="black", linewidth=0.5, zorder=6)
    if len(robust_local):
        robust_local.boundary.plot(ax=ax, color="black", linewidth=0.8, zorder=7)


def plot_class(m, a2):
    fig, ax = plt.subplots(figsize=(11, 9))
    present = [c for c in CLASS_COLORS if c in set(m["final_hazard_screening_class"])]
    m.plot(ax=ax, color=m["final_hazard_screening_class"].map(CLASS_COLORS), edgecolor="#f7f7f7", linewidth=.18)
    overlay_priority_outlines(ax, m)
    base_plot(ax, a2)
    ax.set_title("ADM3 Flood-Hazard Screening Classes (2000–2025)", fontsize=15, weight="bold")
    patches = [Patch(facecolor=CLASS_COLORS[c], label=f"{c} (n={(m['final_hazard_screening_class'] == c).sum()})") for c in present]
    ax.legend(handles=patches, loc="lower left", bbox_to_anchor=(0, -.17), ncol=2, fontsize=8, frameon=False)
    fig.text(.5, .015, "Hazard screening only; exposure and vulnerability are not included.", ha="center", fontsize=9)
    export(fig, "26_01_adm3_final_hazard_screening_map.png")


def plot_class_top10(m, a2):
    top = build_top10_table(m)
    colors = m["final_hazard_screening_class"].map(CLASS_COLORS)
    present = [c for c in CLASS_COLORS if c in set(m["final_hazard_screening_class"])]
    legend = [Patch(facecolor=CLASS_COLORS[c], label=f"{c} (n={(m['final_hazard_screening_class']==c).sum()})") for c in present]
    plot_two_region_poster(
        m, a2, top, colors, legend,
        "ADM3 flood-hazard screening classes · 2000–2025",
        "Hazard score (0–100)", "label_score",
        "26_01b_adm3_final_hazard_screening_map_top10_labels.png",
        "Top 10 by Script 25 hazard-screening score; not humanitarian-action priority.",
        "rank_no", "label_x", "label_y", "label_adm3", "label_adm2",
        outline=True,
    )


# Poster palette: color choices emphasize differentiation, not measured severity tiers.
EXTENT_BINS = [0, 2, 5, 10, 20, 35, 50, 75, 100.00001]
EXTENT_COLORS = ["#fff6d5", "#ffe8a0", "#ffd269", "#ffad42", "#f4772e", "#dc442d", "#ad1738", "#72002d"]
CHANGE_BINS = [-float("inf"), -10, -2, 2, 10, 25, float("inf")]
CHANGE_COLORS = ["#28649d", "#9ac7df", "#e4e5e6", "#ffc9b0", "#ef795b", "#a80e31"]


def plot_two_region_poster(m, a2, top, colors, legend, title, value_title,
                           value_col, filename, note, rank_col, x_col, y_col,
                           name_col, parent_col, outline=False):
    """Two *genuinely local* insets instead of one horizontally compressed inset.

    Top-10 sites are divided geographically into two groups, then numbered with
    exterior callouts. Every featured ADM3 has exactly one numbered location.
    """
    fig = plt.figure(figsize=(18, 12), facecolor="white")
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1.43, 1],
                  height_ratios=[2.0, 2.0, 1.78],
                  left=.028, right=.985, top=.935, bottom=.075,
                  wspace=.055, hspace=.12)
    main_ax = fig.add_subplot(gs[0:2, 0])
    zoom_axes = [fig.add_subplot(gs[i, 1]) for i in range(2)]
    table_ax = fig.add_subplot(gs[2, :])
    for a in [main_ax] + zoom_axes:
        m.plot(ax=a, color=colors, edgecolor="white", linewidth=.17)
        if outline:
            overlay_priority_outlines(a, m)
        a2.boundary.plot(ax=a, color="#535353", linewidth=.6, zorder=5)
        a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([])
    main_ax.axis("off")
    minx, miny, maxx, maxy = m.total_bounds
    sx, sy = maxx-minx, maxy-miny
    main_ax.set_xlim(minx-.018*sx, maxx+.018*sx)
    main_ax.set_ylim(miny-.018*sy, maxy+.018*sy)
    main_ax.set_title("South Sudan · all ADM3", fontsize=15, weight="bold", pad=8)

    # Split geographic points, not by ranking. This preserves spatial meaning.
    extent_x = float(top[x_col].max()-top[x_col].min())
    extent_y = float(top[y_col].max()-top[y_col].min())
    split_col = x_col if extent_x >= extent_y else y_col
    ordered = top.sort_values(split_col, kind="stable")
    groups = [ordered.iloc[:len(ordered)//2], ordered.iloc[len(ordered)//2:]]

    for k, (zoom, group) in enumerate(zip(zoom_axes, groups), start=1):
        x0, y0, x1, y1 = group.total_bounds
        gw, gh = max(x1-x0, 1e-5), max(y1-y0, 1e-5)
        # Include room for callout columns and full polygon footprints.
        x0 -= max(.55*gw, .015*sx); x1 += max(.55*gw, .015*sx)
        y0 -= max(.42*gh, .015*sy); y1 += max(.42*gh, .015*sy)
        # Equal-aspect maps must match the physical subplot shape, otherwise
        # Matplotlib letterboxes a very flat region in the middle of the panel.
        # The extra room becomes meaningful geographic context, not white space.
        panel = zoom.get_position()
        target = (panel.width*fig.get_figwidth())/(panel.height*fig.get_figheight())
        mid_x, mid_y = (x0+x1)/2, (y0+y1)/2
        ww, hh = x1-x0, y1-y0
        if ww/hh < target:
            ww = hh*target
        else:
            hh = ww/target
        x0, x1 = mid_x-ww/2, mid_x+ww/2
        y0, y1 = mid_y-hh/2, mid_y+hh/2
        zoom.set_xlim(x0,x1); zoom.set_ylim(y0,y1)
        zoom.set_title(f"Regional zoom {k} · numbered ADM3", fontsize=14, weight="bold", pad=6)
        for spine in zoom.spines.values():
            spine.set_color("#666666"); spine.set_linewidth(.8)
        main_ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0, fill=False,
                         ec="#222222", lw=1.05, ls="--", zorder=8))

        # Alternate numbers between the left and right margins of each zoom.
        items = group.sort_values(y_col, ascending=False)
        left_items, right_items = items.iloc[::2], items.iloc[1::2]
        for is_right, part in [(False,left_items),(True,right_items)]:
            if len(part)==0: continue
            xx = x1-.055*ww if is_right else x0+.055*ww
            y_positions = [y1-(.17+.66*(i+.5)/len(part))*hh for i in range(len(part))]
            for (_, item), yy in zip(part.iterrows(), y_positions):
                zoom.annotate(str(int(item[rank_col])),
                    xy=(item[x_col],item[y_col]), xytext=(xx,yy),
                    ha="center",va="center",fontsize=10.5,weight="bold",
                    bbox=dict(boxstyle="circle,pad=.30",fc="white",ec="#202020",lw=1.15),
                    arrowprops=dict(arrowstyle="-",lw=.85,color="#444444",shrinkA=11,shrinkB=1),
                    zorder=20,annotation_clip=False)

    # Compact four-column caption table occupies the width beneath the maps.
    table_ax.axis("off")
    ranked = top.sort_values(rank_col)
    rows = []
    for _, r in ranked.iterrows():
        val = float(r[value_col])
        rows.append([str(int(r[rank_col])), str(r[name_col]),str(r[parent_col]),f"{val:.2f}"])
    table = table_ax.table(cellText=rows,
        colLabels=["No.","ADM3", "ADM2",value_title], cellLoc="left",colLoc="left",
        bbox=[.095,.03,.81,.91],colWidths=[.08,.34,.32,.26])
    table.auto_set_font_size(False);table.set_fontsize(10.5)
    for (r,c),cell in table.get_celld().items():
        cell.set_linewidth(.35);cell.set_edgecolor("#c8c8c8")
        if r==0:
            cell.set_facecolor("#e9e9e9"); cell.set_text_props(weight="bold")
        elif r%2==0: cell.set_facecolor("#fafafa")

    fig.legend(handles=legend,loc="lower center",bbox_to_anchor=(.5,.037),
               ncol=min(4,len(legend)),fontsize=9,frameon=False,columnspacing=1.3)
    fig.suptitle(title,fontsize=21,weight="bold",y=.985)
    fig.text(.5,.008,note,ha="center",fontsize=9)
    export(fig,filename)


def plot_extent(m, a2):
    fig, ax = plt.subplots(figsize=(11, 9))
    m.plot(column="recent_combined_mean", cmap="YlOrRd", ax=ax, legend=True,
           vmin=0, vmax=100, edgecolor="#fff", linewidth=.18,
           missing_kwds={"color": "#cccccc", "label": "Missing"},
           legend_kwds={"label": "Average flooded area, 2020–2025 (%)", "shrink": .7})
    base_plot(ax, a2)
    ax.set_title("ADM3 Mean Recent Flood Extent (2020–2025)", fontsize=15, weight="bold")
    export(fig, "26_02_adm3_recent_flood_extent_map.png")


def plot_change(m, a2):
    values = m["combined_change_recent_minus_early"].dropna()
    if values.empty:
        raise ValueError("No flood change values in Script 25 data")
    bound = max(float(values.abs().max()), .1)
    fig, ax = plt.subplots(figsize=(11, 9))
    m.plot(column="combined_change_recent_minus_early", ax=ax, cmap="RdBu_r",
           norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound), legend=True,
           edgecolor="#fff", linewidth=.18,
           missing_kwds={"color": "#cccccc", "label": "Missing"},
           legend_kwds={"label": "Recent minus early flooded area (percentage points)", "shrink": .7})
    base_plot(ax, a2)
    ax.set_title("ADM3 Flood Change: 2020–2025 vs 2000–2019", fontsize=15, weight="bold")
    export(fig, "26_03_adm3_flood_change_map.png")


def plot_robust(m, a2):
    fig, ax = plt.subplots(figsize=(11, 9))
    m.plot(ax=ax, color="#e5e9e9", edgecolor="white", linewidth=.18)
    hot = m[m["robust_flag"]]
    if len(hot):
        hot.plot(ax=ax, color="#b12f40", edgecolor="white", linewidth=.3)
    base_plot(ax, a2)
    ax.set_title(f"Robust Local Flood Amplification (n={len(hot)} ADM3)", fontsize=15, weight="bold")
    ax.legend(handles=[Patch(facecolor="#b12f40", label=f"Robust amplification (n={len(hot)})"),
                       Patch(facecolor="#e5e9e9", label="Other ADM3")], loc="lower left", frameon=False)
    fig.text(.5, .025, "Robustness is across tested thresholds, not a significance or humanitarian-priority test.",
             ha="center", fontsize=8)
    export(fig, "26_04_adm3_robust_local_amplification_map.png")


# Poster palette: color choices emphasize differentiation, not measured severity tiers.
EXTENT_BINS = [0, 2, 5, 10, 20, 35, 50, 75, 100.00001]
EXTENT_COLORS = ["#fff6d5", "#ffe8a0", "#ffd269", "#ffad42", "#f4772e", "#dc442d", "#ad1738", "#72002d"]
CHANGE_BINS = [-float("inf"), -10, -2, 2, 10, 25, float("inf")]
CHANGE_COLORS = ["#28649d", "#9ac7df", "#e4e5e6", "#ffc9b0", "#ef795b", "#a80e31"]




def plot_poster_top10(m, a2, mode):
    configs = {
        "extent": ("recent_combined_mean", "Flood extent (%)", "Recent flood extent · 2020–2025",
                   "26_02b_adm3_recent_flood_extent_top10.png"),
        "change": ("combined_change_recent_minus_early", "Change (pp)",
                   "Flood extent change · 2020–2025 vs 2000–2019",
                   "26_03b_adm3_flood_change_top10.png"),
        "robust": ("final_hazard_score_0_100", "Hazard score (0–100)",
                   "Robust local flood amplification · 45 ADM3",
                   "26_04b_adm3_robust_local_amplification_top10.png"),
    }
    score_col, value_label, title, filename = configs[mode]
    candidates = m[m["robust_flag"]].copy() if mode=="robust" else m.copy()
    top = candidates.dropna(subset=[score_col]).sort_values(
        [score_col,"_key"],ascending=[False,True]).head(10).copy()
    if len(top)<10: raise ValueError(f"Only {len(top)} candidates for {mode}")
    ncol=choose(top,["ADM3_NAME_STD","ADM3_NAME"],"ADM3 name")
    pcol=choose(top,["ADM2_NAME_STD","ADM2_NAME"],"ADM2 name")
    top["plot_rank"]=range(1,11)
    pts=top.geometry.representative_point()
    top["plot_x"],top["plot_y"]=pts.x,pts.y
    top["plot_adm3"]=top[ncol].fillna("Unknown").astype(str)
    top["plot_adm2"]=top[pcol].fillna("Unknown").astype(str)
    top[["plot_rank","_key","plot_adm3","plot_adm2",score_col]].to_csv(
        TAB/f"26_{mode}_top10_labels.csv",index=False)
    if mode=="extent":
        palette=ListedColormap(EXTENT_COLORS)
        shades=palette(BoundaryNorm(EXTENT_BINS,palette.N,clip=True)(m[score_col].clip(0,100).to_numpy()))
        labels=["0–2%","2–5%","5–10%","10–20%","20–35%","35–50%","50–75%","75–100%"]
        legend=[Patch(facecolor=c,label=l) for c,l in zip(EXTENT_COLORS,labels)]
    elif mode=="change":
        palette=ListedColormap(CHANGE_COLORS)
        shades=palette(BoundaryNorm(CHANGE_BINS,palette.N,clip=True)(m[score_col].to_numpy()))
        labels=["< −10 pp","−10 to −2 pp","−2 to +2 pp","+2 to +10 pp","+10 to +25 pp","> +25 pp"]
        legend=[Patch(facecolor=c,label=l) for c,l in zip(CHANGE_COLORS,labels)]
    else:
        shades=m["robust_flag"].map({True:"#a31743",False:"#e1e5e7"})
        legend=[Patch(facecolor="#a31743",label="Robust local amplification (45)"),
                Patch(facecolor="#e1e5e7",label="Other ADM3")]
    note=("Top 10 robust units ranked by Script 25 hazard-screening score; not statistical significance."
          if mode=="robust" else "Top 10 ranked by mapped indicator; not humanitarian-action priority.")
    plot_two_region_poster(m,a2,top,shades,legend,title,value_label,score_col,
                           filename,note,"plot_rank","plot_x","plot_y","plot_adm3","plot_adm2")


def plot_top15(m):
    names = choose(m, ["ADM3_NAME_STD", "ADM3_NAME"], "ADM3 name")
    parents = choose(m, ["ADM2_NAME_STD", "ADM2_NAME"], "ADM2 name")
    top = m.dropna(subset=["final_hazard_score_0_100"]).nlargest(15, "final_hazard_score_0_100").copy()
    top["display"] = top[names].astype(str) + " — " + top[parents].astype(str)
    top = top.iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(top["display"], top["final_hazard_score_0_100"], color="#a3303d")
    ax.set_xlim(0, 105)
    ax.set_xlabel("Script 25 relative hazard-screening score (0–100)")
    ax.set_title("Top 15 ADM3 by Flood-Hazard Screening Score", fontsize=15, weight="bold")
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    export(fig, "26_05_top15_adm3_hazard_scores.png")
    top[["display", "final_hazard_score_0_100", "final_hazard_screening_class"]].iloc[::-1].to_csv(
        TAB / "26_top15_adm3_hazard_scores.csv", index=False)


def main():
    for p in [FIG, TAB, QC]:
        p.mkdir(parents=True, exist_ok=True)
    m, a2, join_method = prepare()
    m.drop(columns="geometry").to_csv(TAB / "26_adm3_poster_map_data.csv", index=False)
    plot_class(m, a2)
    plot_class_top10(m, a2)
    plot_extent(m, a2)
    plot_change(m, a2)
    plot_robust(m, a2)
    plot_top15(m)
    for mode in ("extent", "change", "robust"):
        plot_poster_top10(m, a2, mode)
    report = [
        "SCRIPT 26 — ADM3 POSTER FIGURES QC",
        f"Input: {INPUT_CSV}",
        f"ADM3 boundary: {ADM3_GEOJSON}",
        f"ADM2 boundary: {ADM2_GEOJSON}",
        f"Join method: {join_method}",
        f"ADM3 rows: {len(m)}",
        f"Unique match keys: {m['_key'].nunique()}",
        f"Robust amplification units: {int(m['robust_flag'].sum())}",
        f"Missing extent: {int(m['recent_combined_mean'].isna().sum())}",
        f"Missing change: {int(m['combined_change_recent_minus_early'].isna().sum())}",
        f"Top labelled units: {TOP_N}",
        "Class counts:",
        m["final_hazard_screening_class"].value_counts().to_string(),
        "Interpretation: Script 25 hazard screening only, not humanitarian priority.",
    ]
    (QC / "26_poster_map_qc.txt").write_text("\n".join(report), encoding="utf-8")
    print("Script 26 complete:", FIG)


if __name__ == "__main__":
    main()
