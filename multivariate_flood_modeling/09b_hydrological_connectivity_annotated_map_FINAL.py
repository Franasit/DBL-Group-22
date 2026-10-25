#!/usr/bin/env python3
"""
09b_hydrological_connectivity_annotated_map.py

Create an annotated hydrological connectivity map for South Sudan
using outputs from Script 09 plus HydroRIVERS and hydrology source points.

Main purpose:
- show South Sudan administrative / basin context
- show major rivers
- show lake-level source stations
- show DFO discharge stations
- annotate directly on the figure what triangles / dots / river lines / coloured polygons mean

Expected upstream data:
1) Script 09 already run:
   processed_data/multivariate_flood_modeling/09_major_basin_mapping/
       adm3_dominant_basins.gpkg
       adm2_dominant_basins.gpkg

2) Hydrology source locations:
   processed_data/multivariate_flood_modeling/06c_hydrological_spatial_mapping/
       01_hydrology_source_locations.csv

3) HydroRIVERS geodatabase:
   raw_data/HydroRIVERS_v10_af/HydroRIVERS_v10_af.gdb

4) Administrative boundaries (only needed if GPKG output from script 09 is absent):
   raw_data/administrative_boundaries/ssd_admin2.geojson
   raw_data/administrative_boundaries/ssd_admin3.geojson

Outputs:
   processed_data/multivariate_flood_modeling/09b_hydrological_connectivity_annotated/
       annotated_adm3_map.png
       annotated_adm3_map.pdf
       annotated_adm2_map.png
       annotated_adm2_map.pdf

Notes:
- The figure is explanatory / cartographic. It does NOT prove hydrological causation.
- Orange triangles = DFO discharge sources
- Purple circles = lake water-level sources
- Blue lines = HydroRIVERS major rivers
- Coloured polygons = dominant HydroBASINS overlap by area
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

from matplotlib.offsetbox import AnchoredText
from shapely.geometry import Point

GEO_CRS = "EPSG:4326"
AREA_CRS = "EPSG:6933"
MAP_BG = "#FCFBFE"

POLY_PALETTE = [
    "#D8E7F8", "#E7D8F3", "#D5F0EC", "#EFD8F4",
    "#CFE6F8", "#E9D6EA", "#DCE6F4", "#D7F0F2",
    "#E2D2F7", "#CFE1E9", "#F1D8E7", "#DADDF2",
]

COLOR_RIVER = "#2E8CCB"
COLOR_DFO = "#E17B39"
COLOR_LAKE = "#7546B7"
COLOR_BOUNDARY = "#34303A"
COLOR_TEXT = "#2F2A33"

plt.rcParams.update({
    "font.size": 10,
    "font.family": "DejaVu Sans",
})


def project_root() -> Path:
    here = Path(__file__).resolve()
    if here.parent.name == "multivariate_flood_modeling":
        return here.parents[2]
    return Path.cwd()


def resolved(root: Path, p: str) -> Path:
    x = Path(p).expanduser()
    return x if x.is_absolute() else root / x


def read_plot_layer(root: Path, level: str, mapping_dir: Path) -> gpd.GeoDataFrame:
    gpkg = mapping_dir / f"{level.lower()}_dominant_basins.gpkg"
    if gpkg.exists():
        g = gpd.read_file(gpkg)
        if g.crs is None:
            raise ValueError(f"Missing CRS in {gpkg}")
        return g.to_crs(AREA_CRS)

    # Fallback: load raw boundaries (without basin columns)
    warnings.warn(
        f"{gpkg} not found. Falling back to raw administrative boundaries only; "
        "coloured basin polygons will not be available."
    )
    raw = root / f"raw_data/administrative_boundaries/ssd_{level.lower()}.geojson"
    if not raw.exists():
        raise FileNotFoundError(gpkg)
    g = gpd.read_file(raw)
    if g.crs is None:
        raise ValueError(f"Missing CRS in {raw}")
    g = g.to_crs(AREA_CRS)
    # create placeholders
    g["dominant_hybas_id"] = np.nan
    return g


def read_adm2_outline(root: Path) -> gpd.GeoDataFrame:
    p = root / "raw_data/administrative_boundaries/ssd_admin2.geojson"
    if not p.exists():
        raise FileNotFoundError(p)
    g = gpd.read_file(p)
    if g.crs is None:
        raise ValueError(f"Missing CRS in {p}")
    return g.to_crs(AREA_CRS)


def read_sources(root: Path) -> gpd.GeoDataFrame:
    p = root / "processed_data/multivariate_flood_modeling/06c_hydrological_spatial_mapping/01_hydrology_source_locations.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p, dtype={"source_id": "string"})
    required = {"source_type", "source_id", "longitude", "latitude"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns in source file: {required - set(df.columns)}")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df = df.dropna(subset=["longitude", "latitude"]).copy()
    df["source_type"] = df["source_type"].astype(str).str.lower().str.strip()
    g = gpd.GeoDataFrame(df,
                         geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
                         crs=GEO_CRS)
    return g.to_crs(AREA_CRS)


def read_rivers(root: Path, bounds_geom, stream_order_min: int = 5) -> gpd.GeoDataFrame:
    river_root = root / "raw_data/HydroRIVERS_v10_af"
    gdb = river_root / "HydroRIVERS_v10_af.gdb"
    if not gdb.exists():
        matches = sorted(river_root.rglob("*.gdb")) if river_root.exists() else []
        if not matches:
            raise FileNotFoundError(f"No HydroRIVERS geodatabase found in {river_root}")
        gdb = matches[0]

    bounds_wgs = gpd.GeoSeries([bounds_geom], crs=AREA_CRS).to_crs(GEO_CRS).total_bounds
    rivers = gpd.read_file(gdb, bbox=tuple(bounds_wgs))
    if rivers.crs is None:
        raise ValueError(f"Missing CRS in {gdb}")
    order_col = None
    for c in ["ORD_STRA", "ORD_FLOW", "ORD_CLAS"]:
        if c in rivers.columns:
            order_col = c
            break
    if order_col:
        rivers[order_col] = pd.to_numeric(rivers[order_col], errors="coerce")
        rivers = rivers[rivers[order_col].ge(stream_order_min)].copy()
    return rivers.to_crs(AREA_CRS)


def assign_polygon_colors(g: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict]:
    out = g.copy()
    if "dominant_hybas_id" not in out.columns or out["dominant_hybas_id"].dropna().empty:
        out["plot_color"] = "#EAE6EE"
        return out, {}

    ids = sorted(pd.Series(out["dominant_hybas_id"].dropna().unique()).astype("int64"))
    cmap = {bid: POLY_PALETTE[i % len(POLY_PALETTE)] for i, bid in enumerate(ids)}
    out["plot_color"] = out["dominant_hybas_id"].map(cmap).fillna("#EAE6EE")
    return out, cmap


def add_north_arrow(ax, xy=(0.94, 0.91), size=0.07):
    x, y = xy
    ax.annotate(
        "N",
        xy=(x, y),
        xytext=(x, y - size * 0.55),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=COLOR_TEXT,
        arrowprops=dict(arrowstyle="-|>", lw=1.4, color=COLOR_TEXT),
    )


def add_scalebar(ax, length_km=200, location=(0.74, 0.06)):
    x0, y0 = location
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    width = xmax - xmin
    height = ymax - ymin

    # approximate length in projected CRS (metres)
    length_m = length_km * 1000.0
    start_x = xmin + width * x0
    start_y = ymin + height * y0
    end_x = start_x + length_m

    ax.plot([start_x, end_x], [start_y, start_y], color=COLOR_TEXT, lw=3, solid_capstyle="butt", zorder=20)
    for frac in [0, 0.5, 1]:
        xx = start_x + (end_x - start_x) * frac
        ax.plot([xx, xx], [start_y - height * 0.006, start_y + height * 0.006], color=COLOR_TEXT, lw=1.3, zorder=20)

    ax.text(start_x, start_y + height * 0.012, "0", fontsize=8.5, ha="center", color=COLOR_TEXT)
    ax.text(start_x + (end_x - start_x) * 0.5, start_y + height * 0.012, f"{int(length_km/2)}", fontsize=8.5, ha="center", color=COLOR_TEXT)
    ax.text(end_x, start_y + height * 0.012, f"{length_km} km", fontsize=8.5, ha="center", color=COLOR_TEXT)


def choose_representative_point(gdf: gpd.GeoDataFrame, strategy: str = "first"):
    if gdf.empty:
        return None
    if strategy == "northmost":
        idx = gdf.geometry.y.idxmax()
        return gdf.loc[idx]
    if strategy == "southmost":
        idx = gdf.geometry.y.idxmin()
        return gdf.loc[idx]
    if strategy == "eastmost":
        idx = gdf.geometry.x.idxmax()
        return gdf.loc[idx]
    if strategy == "westmost":
        idx = gdf.geometry.x.idxmin()
        return gdf.loc[idx]
    return gdf.iloc[0]


def line_midpoint(geom):
    try:
        return geom.interpolate(0.5, normalized=True)
    except Exception:
        return geom.representative_point()


def add_callout(ax, anchor_xy, text_xy_axes, text, color, box_fc="white", align="left"):
    ha = "left" if align == "left" else "right"
    ax.annotate(
        text,
        xy=anchor_xy,
        xycoords="data",
        xytext=text_xy_axes,
        textcoords="axes fraction",
        ha=ha,
        va="center",
        fontsize=10,
        color=COLOR_TEXT,
        bbox=dict(boxstyle="round,pad=0.35", fc=box_fc, ec=color, lw=1.2, alpha=0.97),
        arrowprops=dict(
            arrowstyle="->",
            color=color,
            lw=1.35,
            shrinkA=4,
            shrinkB=4,
            connectionstyle="arc3,rad=0.18",
        ),
        zorder=50,
    )


def plot_annotated_map(root: Path, level: str, mapping_dir: Path, output_dir: Path, stream_order_min: int = 5):
    layer = read_plot_layer(root, level, mapping_dir)
    layer, color_map = assign_polygon_colors(layer)
    adm2_outline = read_adm2_outline(root)
    country = adm2_outline.dissolve()
    bounds_geom = country.geometry.iloc[0].buffer(250000)  # small outer context
    rivers = read_rivers(root, bounds_geom, stream_order_min=stream_order_min)
    sources = read_sources(root)
    lakes = sources[sources["source_type"].eq("lake")].copy()
    dfo = sources[sources["source_type"].eq("dfo")].copy()

    fig = plt.figure(figsize=(16.0, 10.0), facecolor=MAP_BG)
    # Separate map and symbol-key axes to avoid overlays.
    ax = fig.add_axes([0.035, 0.100, 0.680, 0.805])
    info = fig.add_axes([0.742, 0.115, 0.235, 0.765])
    info.axis("off")

    # Plot basin-dominant polygons
    if "plot_color" in layer.columns:
        for clr, sub in layer.groupby("plot_color"):
            sub.plot(ax=ax, color=clr, edgecolor="white", linewidth=0.22, alpha=0.92, zorder=3)
    else:
        layer.plot(ax=ax, color="#ECE8F0", edgecolor="white", linewidth=0.2, zorder=3)

    # Boundaries
    if level.upper() == "ADM3":
        layer.boundary.plot(ax=ax, color="#B8B0C0", linewidth=0.17, alpha=0.75, zorder=4)
        adm2_outline.boundary.plot(ax=ax, color=COLOR_BOUNDARY, linewidth=0.85, alpha=0.95, zorder=5)
    else:
        layer.boundary.plot(ax=ax, color=COLOR_BOUNDARY, linewidth=0.85, alpha=0.95, zorder=5)

    # Rivers
    if not rivers.empty:
        rivers.plot(ax=ax, color=COLOR_RIVER, linewidth=0.9, alpha=0.95, zorder=6)

    # Source points
    if not dfo.empty:
        dfo.plot(ax=ax, marker="^", color=COLOR_DFO, markersize=80,
                 edgecolor="white", linewidth=0.6, zorder=8)
    if not lakes.empty:
        lakes.plot(ax=ax, marker="o", color=COLOR_LAKE, markersize=80,
                   edgecolor="white", linewidth=0.6, zorder=8)

    # Layout and axis
    ax.set_aspect("equal")
    ax.set_axis_off()

    title = "South Sudan: HydroBASINS / HydroRIVERS / hydrology sources"
    subtitle = f"Annotated review map ({level.upper()})"
    fig.suptitle(title, fontsize=19, fontweight="bold", y=0.975, x=0.5, color=COLOR_TEXT)
    fig.text(0.5, 0.935, subtitle, ha="center", fontsize=11.5, color="#5B5562")

    # North arrow & scalebar
    add_north_arrow(ax, xy=(0.95, 0.92), size=0.08)
    add_scalebar(ax, length_km=200, location=(0.72, 0.055))

    # Direct symbol legend: draw the symbols themselves beside their labels.
    # No "purple circle" / "orange triangle" prose or duplicate bottom legend.
    info.set_xlim(0, 1)
    info.set_ylim(0, 1)
    info.add_patch(mpatches.FancyBboxPatch(
        (0.015, 0.28), 0.970, 0.675,
        boxstyle="round,pad=0.012,rounding_size=0.020",
        transform=info.transAxes, facecolor="white", edgecolor="#DAD4E2",
        linewidth=1.0, zorder=0
    ))
    info.text(0.07, 0.905, "Map legend", transform=info.transAxes,
              fontsize=16, fontweight="bold", color=COLOR_TEXT, va="top")

    legend_rows = [
        ("dfo", 0.795, "DFO discharge station"),
        ("lake", 0.705, "Lake water-level station"),
        ("river", 0.615, "HydroRIVERS major rivers"),
        ("basin", 0.525, "HydroBASINS / basin overlap"),
        ("border", 0.435, "South Sudan boundary"),
        ("adm", 0.345, f"{level.upper()} boundaries"),
    ]
    for kind, y, label in legend_rows:
        if kind == "dfo":
            info.scatter([0.125], [y], marker="^", s=145, color=COLOR_DFO,
                         edgecolors="white", linewidths=0.8, zorder=3)
        elif kind == "lake":
            info.scatter([0.125], [y], marker="o", s=140, color=COLOR_LAKE,
                         edgecolors="white", linewidths=0.8, zorder=3)
        elif kind == "river":
            info.plot([0.07, 0.19], [y, y], color=COLOR_RIVER, lw=2.8,
                      solid_capstyle="round", zorder=3)
        elif kind == "basin":
            info.add_patch(mpatches.Rectangle((0.085, y-0.022), 0.083, 0.044,
                          facecolor="#D8E7F8", edgecolor="#C18AE0", lw=0.9))
        else:
            info.plot([0.075, 0.185], [y, y], color=COLOR_BOUNDARY if kind == "border" else "#B8B0C0",
                      lw=2.6 if kind == "border" else 1.2, zorder=3)
        info.text(0.235, y, label, fontsize=10.4, color=COLOR_TEXT,
                  va="center", ha="left")

    info.text(0.04, 0.225, "Interpretation note", fontweight="bold",
              fontsize=11.1, color=COLOR_TEXT, va="top")
    info.text(0.04, 0.185,
              "Basin colours represent the dominant\n"
              "HydroBASINS overlap for each ADM.\n"
              "Source-to-region connections remain\n"
              "UNVERIFIED network candidates.",
              fontsize=9.8, color="#57505F", va="top", linespacing=1.45)

    fig.text(0.04, 0.032,
             "Visual comparison only. River flow connectivity and prediction-time availability require separate verification.",
             fontsize=9.2, color="#4E4754")

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"annotated_{level.lower()}_map.png"
    pdf = output_dir / f"annotated_{level.lower()}_map.pdf"
    fig.savefig(png, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    fig.savefig(pdf, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main():
    parser = argparse.ArgumentParser(description="Create an annotated hydrological connectivity map.")
    parser.add_argument("--root", default=str(project_root()))
    parser.add_argument("--mapping-dir", default="processed_data/multivariate_flood_modeling/09_major_basin_mapping")
    parser.add_argument("--output", default="processed_data/multivariate_flood_modeling/09b_hydrological_connectivity_annotated")
    parser.add_argument("--level", default="ADM3", choices=["ADM2", "ADM3", "both"],
                        help="Map level to draw.")
    parser.add_argument("--stream-order-min", type=int, default=5,
                        help="Minimum HydroRIVERS order used for display.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    mapping_dir = resolved(root, args.mapping_dir)
    output_dir = resolved(root, args.output)

    levels = ["ADM2", "ADM3"] if args.level == "both" else [args.level]
    results = []
    for lvl in levels:
        png, pdf = plot_annotated_map(root, lvl, mapping_dir, output_dir, stream_order_min=args.stream_order_min)
        results.append((lvl, png, pdf))
        print(f"{lvl}:")
        print(f"  PNG -> {png}")
        print(f"  PDF -> {pdf}")

    print("\nDone.")


if __name__ == "__main__":
    main()
