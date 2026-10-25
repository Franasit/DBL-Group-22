#!/usr/bin/env python3
"""09_major_basin_adm2_adm3_mapping.py

JBG060 South Sudan: administrative boundaries x HydroBASINS,
major-river context, and candidate monitoring links.

Default project files:
  raw_data/hybas_af_lev01-12_v1c/hybas_af_lev08_v1c.shp
  raw_data/HydroRIVERS_v10_af/HydroRIVERS_v10_af.gdb
  raw_data/administrative_boundaries/ssd_admin2.geojson
  raw_data/administrative_boundaries/ssd_admin3.geojson
  processed_data/multivariate_flood_modeling/
      06c_hydrological_spatial_mapping/01_hydrology_source_locations.csv
      06e_river_basin_connectivity/04_downstream_admin_candidates_REVIEW.csv

IMPORTANT SCIENTIFIC CONSTRAINTS
--------------------------------
* This program computes polygon area overlap, which IS a measurable spatial
  relationship. It does NOT automatically prove flood-wave propagation.
* HydroBASINS polygons do not include trustworthy river-system names such as
  "White Nile", "Sobat/Pibor", or "Bahr el Ghazal" in the standard attributes.
  Consequently all basin names remain UNCLASSIFIED unless an optional REVIEWED
  ID-to-system crosswalk is supplied.
* HydroRIVERS order is used to thin a display layer, not to prove hydrological
  connectivity between a source and an ADM3 area.
* Unreviewed 06e candidate connections remain explicitly UNVERIFIED.

Run:
  python scripts/multivariate_flood_modeling/09_major_basin_adm2_adm3_mapping.py

Optional:
  --basin-level 08
  --system-crosswalk path/to/reviewed_basin_systems.csv

Crosswalk columns:
  HYBAS_ID,hydrological_system,evidence_reference,review_status
  (..., White Nile / Sudd, reference, APPROVED)
Only APPROVED rows with a nonempty evidence reference and system label
are accepted. This is a user-reviewed geographic classification, not inferred.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

PROJECT_BASE = Path("processed_data/multivariate_flood_modeling")
GEO_CRS = "EPSG:4326"
AREA_CRS = "EPSG:6933"
COLOR_BLUE = "#7EC9F2"
COLOR_LILAC = "#C18AE0"
COLOR_PURPLE = "#7546B7"
MAP_BG = "#FCFBFE"
SYSTEM_COLORS = [
    "#7EC9F2", "#C18AE0", "#7546B7", "#E5B8ED",
    "#71B4BD", "#B5A4DE", "#CF8FBD", "#7C9DC5"
]


def project_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[2] if here.parent.name == "multivariate_flood_modeling" else Path.cwd()


def resolved(root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else root / candidate


def pick_column(df, options: list[str], what: str) -> str:
    lookup = {str(c).lower(): c for c in df.columns}
    for option in options:
        if option.lower() in lookup:
            return lookup[option.lower()]
    raise ValueError(f"Cannot identify {what}; available columns: {list(df.columns)}")


def load_admin(path: Path, level: str) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    g = gpd.read_file(path)
    if g.crs is None:
        raise ValueError(f"Missing CRS in {path}")
    lvl = level.lower()
    key = pick_column(g,
        [f"{lvl}_pcode", f"{lvl}_id", f"admin{lvl[-1]}Pcod", "adm_id"],
        f"{level} Pcode")
    name = next((c for c in [
        f"{lvl}_name", f"{lvl}_en", f"admin{lvl[-1]}Name", f"{lvl}_NAME"
    ] if c in g.columns), None)
    parent = next((c for c in [
        "adm2_name", "ADM2_NAME", "adm2_en", "admin2Name", "flood__ADM2_NAME"
    ] if c in g.columns), None) if level == "ADM3" else None
    result = g[[key, "geometry"]].copy()
    result["adm_id"] = result[key].astype("string").str.strip()
    result["admin_name"] = g[name].astype("string") if name else result["adm_id"]
    result["parent_adm2_name"] = g[parent].astype("string") if parent else pd.NA
    result = result[["adm_id", "admin_name", "parent_adm2_name", "geometry"]].dropna(subset=["geometry"])
    if result["adm_id"].isna().any() or result["adm_id"].duplicated().any():
        raise ValueError(f"Missing or duplicate administrative IDs in {path}")
    return result.to_crs(AREA_CRS)


def load_basins(path: Path, admin_bounds: tuple, pad_km: int = 100) -> tuple[gpd.GeoDataFrame, int]:
    if not path.exists():
        raise FileNotFoundError(path)
    # Spatially filter early using a WGS84 bounding box with padding.
    from shapely.geometry import box
    bbox_metric = gpd.GeoSeries([box(*admin_bounds)], crs=AREA_CRS)
    buffered = bbox_metric.buffer(pad_km * 1000).to_crs(GEO_CRS).total_bounds
    basins = gpd.read_file(path, bbox=tuple(buffered))
    if basins.empty:
        raise ValueError("HydroBASINS returned zero polygons for South Sudan")
    if basins.crs is None:
        raise ValueError(f"Missing CRS in {path}")
    required = {"HYBAS_ID", "NEXT_DOWN"}
    if not required.issubset(basins):
        raise ValueError(f"HydroBASINS missing columns: {required - set(basins.columns)}")
    basins["HYBAS_ID"] = pd.to_numeric(basins["HYBAS_ID"], errors="raise").astype("int64")
    if basins["HYBAS_ID"].duplicated().any():
        raise ValueError("Duplicate HYBAS_ID found in basin subset")
    return basins.to_crs(AREA_CRS), len(basins)


def calculate_overlaps(adm: gpd.GeoDataFrame, basins: gpd.GeoDataFrame,
                       level: str, min_overlap_percent: float = 0.05) -> pd.DataFrame:
    """Compute true intersection area divided by each ADM polygon's full area."""
    units = adm.copy()
    units["adm_area_km2"] = units.geometry.area / 1e6
    basins_small = basins[["HYBAS_ID", "NEXT_DOWN", "geometry"]].copy()
    # Match candidates by spatial index without exhaustive cross join.
    possible = gpd.sjoin(units, basins_small, predicate="intersects", how="inner")
    if possible.empty:
        raise ValueError(f"{level}: no basin/ADM intersections")

    basin_lookup = basins_small.geometry
    records = []
    for _, r in possible.iterrows():
        bg = basin_lookup.loc[r["index_right"]]
        intersection = r.geometry.intersection(bg)
        if intersection.is_empty:
            continue
        area_km2 = float(intersection.area / 1e6)
        if area_km2 <= 0:
            continue
        admin_area = float(r["adm_area_km2"])
        frac = area_km2 / admin_area if admin_area > 0 else np.nan
        if not np.isfinite(frac) or frac * 100 < min_overlap_percent:
            continue
        records.append({
            "level": level,
            "adm_id": str(r["adm_id"]),
            "admin_name": str(r["admin_name"]),
            "parent_adm2_name": (None if pd.isna(r["parent_adm2_name"]) else str(r["parent_adm2_name"])),
            "HYBAS_ID": int(r["HYBAS_ID"]),
            "NEXT_DOWN": int(r["NEXT_DOWN"]) if pd.notna(r["NEXT_DOWN"]) else None,
            "overlap_area_km2": area_km2,
            "adm_area_km2": admin_area,
            "adm_area_fraction": frac,
            "adm_area_percent": frac * 100,
        })
    if not records:
        raise ValueError(f"{level}: no overlaps above minimum percentage.")
    result = pd.DataFrame.from_records(records)
    result = result.sort_values(["adm_id", "adm_area_percent"], ascending=[True, False])
    result["basin_rank_within_adm"] = result.groupby("adm_id").cumcount() + 1
    return result


def apply_crosswalk(overlaps: pd.DataFrame, crosswalk_path: Path | None) -> pd.DataFrame:
    result = overlaps.copy()
    result["hydrological_system"] = "UNCLASSIFIED"
    result["system_review_status"] = "NOT_REVIEWED"
    if crosswalk_path is None:
        return result
    if not crosswalk_path.exists():
        raise FileNotFoundError(crosswalk_path)
    mapping = pd.read_csv(crosswalk_path)
    required = {"HYBAS_ID", "hydrological_system", "evidence_reference", "review_status"}
    if not required.issubset(mapping):
        raise ValueError(f"Crosswalk missing {required - set(mapping.columns)}")
    mapping["HYBAS_ID"] = pd.to_numeric(mapping["HYBAS_ID"], errors="raise").astype("int64")
    mapping["review_status"] = mapping["review_status"].astype(str).str.upper().str.strip()
    approved = mapping[mapping["review_status"].eq("APPROVED")].copy()
    approved = approved[
        approved["hydrological_system"].notna()
        & approved["evidence_reference"].astype(str).str.strip().str.len().ge(8)
    ]
    if approved["HYBAS_ID"].duplicated().any():
        raise ValueError("Approved crosswalk has duplicate basin IDs")
    approved = approved[["HYBAS_ID", "hydrological_system"]]
    result = result.merge(approved, on="HYBAS_ID", how="left", suffixes=("", "_approved"),
                          validate="many_to_one")
    present = result["hydrological_system_approved"].notna()
    result.loc[present, "hydrological_system"] = result.loc[present, "hydrological_system_approved"].astype(str)
    result.loc[present, "system_review_status"] = "APPROVED"
    return result.drop(columns=["hydrological_system_approved"])


def summaries(overlaps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    main = overlaps[overlaps["basin_rank_within_adm"].eq(1)].copy()
    counts = overlaps.groupby("adm_id").agg(
        n_overlapping_basins=("HYBAS_ID", "nunique"),
        sum_reported_overlap_percent=("adm_area_percent", "sum"),
    ).reset_index()
    main = main.merge(counts, on="adm_id", how="left")
    main = main.rename(columns={
        "HYBAS_ID": "dominant_hybas_id",
        "adm_area_percent": "dominant_basin_percent",
        "hydrological_system": "dominant_system_name",
    })
    by_system = overlaps.groupby([
        "level", "adm_id", "admin_name", "hydrological_system"
    ], dropna=False, as_index=False).agg(
        total_overlap_km2=("overlap_area_km2", "sum"),
        share_admin_percent=("adm_area_percent", "sum"),
        basin_count=("HYBAS_ID", "nunique")
    )
    return main, by_system


def read_sources(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        print(f"Optional stations file not found: {path}")
        return gpd.GeoDataFrame(columns=["source_type", "source_id", "geometry"],
                                geometry="geometry", crs=GEO_CRS)
    df = pd.read_csv(path, dtype={"source_id": "string"})
    if not {"source_type", "source_id", "longitude", "latitude"}.issubset(df):
        raise ValueError(f"Source locations missing columns: {set(['source_type','source_id','longitude','latitude'])-set(df)}")
    for c in ["longitude", "latitude"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["longitude"].between(-180, 180) & df["latitude"].between(-90, 90)].copy()
    df["source_type"] = df["source_type"].astype(str).str.lower().str.strip()
    df = df.drop_duplicates(["source_type", "source_id"])
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=GEO_CRS)


def read_candidates(path: Path) -> pd.DataFrame:
    cols = ["level", "adm_id", "source_type", "source_id", "downstream_hops"]
    if not path.exists():
        print(f"Optional candidate connections absent: {path}")
        return pd.DataFrame(columns=cols)
    d = pd.read_csv(path, dtype={"adm_id": "string", "source_id": "string"})
    if not set(cols).issubset(d):
        raise ValueError(f"06e candidates missing: {set(cols) - set(d)}")
    d = d[cols].copy()
    d["level"] = d["level"].astype(str).str.upper()
    d["source_type"] = d["source_type"].astype(str).str.lower()
    d["adm_id"] = d["adm_id"].astype("string").str.strip()
    return d.drop_duplicates(["level", "adm_id", "source_type", "source_id"])


def attach_candidate_summary(dom: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        dom["candidate_source_count"] = 0
        dom["candidate_lakes"] = ""
        dom["candidate_dfo"] = ""
        dom["candidate_status"] = "NOT_TESTED"
        return dom
    rows = []
    for (level, aid), part in candidates.groupby(["level", "adm_id"]):
        lakes = sorted(set(part.loc[part.source_type.eq("lake"), "source_id"].dropna().astype(str)))
        dfo = sorted(set(part.loc[part.source_type.eq("dfo"), "source_id"].dropna().astype(str)))
        rows.append({
            "level": level, "adm_id": aid,
            "candidate_source_count": len(lakes) + len(dfo),
            "candidate_lakes": "; ".join(lakes),
            "candidate_dfo": "; ".join(dfo),
        })
    dom = dom.merge(pd.DataFrame(rows), on=["level", "adm_id"], how="left")
    dom["candidate_source_count"] = dom["candidate_source_count"].fillna(0).astype(int)
    for c in ["candidate_lakes", "candidate_dfo"]:
        dom[c] = dom[c].fillna("")
    dom["candidate_status"] = np.where(dom["candidate_source_count"].gt(0),
                                        "UNVERIFIED_NETWORK_CANDIDATE", "NO_NETWORK_CANDIDATE")
    return dom


def load_rivers(path: Path, bounding_polygon, stream_order_min: int) -> gpd.GeoDataFrame:
    if not path.exists():
        print(f"Optional river data missing: {path}")
        return gpd.GeoDataFrame(geometry=[], crs=AREA_CRS)
    bounds = gpd.GeoSeries([bounding_polygon], crs=AREA_CRS).to_crs(GEO_CRS).total_bounds
    g = gpd.read_file(path, bbox=tuple(bounds))
    if g.crs is None:
        raise ValueError(f"HydroRIVERS dataset has no CRS: {path}")
    order_col = next((x for x in ["ORD_STRA", "ORD_FLOW", "ORD_CLAS"] if x in g.columns), None)
    if order_col:
        g[order_col] = pd.to_numeric(g[order_col], errors="coerce")
        g = g[g[order_col].ge(stream_order_min)]
    else:
        print("WARNING: No known river-order column. Skipping river layer rather than drawing all streams.")
        return gpd.GeoDataFrame(geometry=[], crs=AREA_CRS)
    return g.to_crs(AREA_CRS)


def draw_map(adm: gpd.GeoDataFrame, dominant: pd.DataFrame,
             basins: gpd.GeoDataFrame, rivers: gpd.GeoDataFrame,
             stations: gpd.GeoDataFrame, level: str,
             out_png: Path, out_gpkg: Path) -> dict:
    plot_df = adm.merge(dominant[[
        "adm_id", "dominant_hybas_id", "dominant_basin_percent",
        "dominant_system_name", "system_review_status", "candidate_source_count"
    ]], on="adm_id", how="left", validate="one_to_one")
    plot_df.to_file(out_gpkg, driver="GPKG")
    # Distinguish validated basin names only; otherwise color by coarse HYBAS_ID
    ids = sorted(plot_df["dominant_hybas_id"].dropna().astype("int64").unique())
    colors = {int(k): SYSTEM_COLORS[i % len(SYSTEM_COLORS)] for i, k in enumerate(ids)}
    fig = plt.figure(figsize=(18, 10), facecolor=MAP_BG)
    ax = fig.add_axes([0.045, 0.15, 0.72, 0.75])
    inf = fig.add_axes([0.79, 0.15, 0.19, 0.75]); inf.set_axis_off()

    for bid, sub in plot_df.dropna(subset=["dominant_hybas_id"]).groupby("dominant_hybas_id"):
        sub.plot(ax=ax, color=colors[int(bid)], alpha=0.62,
                 edgecolor="white", linewidth=0.25)
    if plot_df["dominant_hybas_id"].isna().any():
        plot_df[plot_df["dominant_hybas_id"].isna()].plot(ax=ax, color="#E6E6E6",
                                                         edgecolor="white", linewidth=0.2)
    adm.dissolve().boundary.plot(ax=ax, color="#37313E", linewidth=1.15)
    if level == "ADM3":
        adm.boundary.plot(ax=ax, color="#88818B", linewidth=0.2, alpha=0.45)
    if not rivers.empty:
        rivers.plot(ax=ax, color="#1F71A5", linewidth=0.65, alpha=0.8, zorder=5)
    if not stations.empty:
        pts = stations.to_crs(AREA_CRS)
        for typ, shape, color in (("lake", "o", "#7546B7"), ("dfo", "^", "#E17B39")):
            subset = pts[pts.source_type.eq(typ)]
            if not subset.empty:
                subset.plot(ax=ax, marker=shape, color=color, markersize=55,
                            edgecolor="white", linewidth=0.55, zorder=7)
    ax.set_axis_off()
    ax.set_title(f"{level} — dominant HydroBASINS Level 08 polygon by area", fontsize=15, weight="bold")
    inf.text(0, 0.95, "Interpretation", fontsize=14, weight="bold")
    inf.text(0, 0.85,
             "Colors distinguish basin IDs,\nnot verified named river systems.\n\n"
             "ADM units can overlap several\nbasins; CSV tables retain all shares.\n\n"
             "Blue lines: selected major rivers\n"
             "Purple dots: lake sources\n"
             "Orange triangles: DFO stations\n\n"
             "Source-to-ADM links from 06e\nremain UNVERIFIED.",
             fontsize=10, va="top", linespacing=1.55)
    fig.suptitle("South Sudan | Administrative Areas and Hydrological Systems",
                 fontsize=20, weight="bold", y=0.97)
    fig.text(.05, .075,
             "Spatial overlap ≠ hydrological causation. Dominant basin is a geographic descriptor; "
             "use approved catchment mapping for named systems and predictive links.",
             fontsize=9)
    fig.savefig(out_png, dpi=250, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"mapped_polygons": int(plot_df["dominant_hybas_id"].notna().sum()),
            "total_polygons": int(len(plot_df)), "distinct_dominant_basins": len(ids)}


def main() -> None:
    p = argparse.ArgumentParser(description="Map South Sudan ADM2/ADM3 overlaps with HydroBASINS.")
    p.add_argument("--root", default=str(project_root()))
    p.add_argument("--basin-level", type=int, default=8)
    p.add_argument("--stream-order-min", type=int, default=5)
    p.add_argument("--min-overlap-percent", type=float, default=0.05)
    p.add_argument("--system-crosswalk", default=None,
                   help="Reviewed CSV mapping HYBAS_ID to named hydrological system.")
    p.add_argument("--output", default=str(PROJECT_BASE / "09_major_basin_mapping"))
    args = p.parse_args()
    root = Path(args.root).expanduser().resolve()
    level = f"{args.basin_level:02d}"
    out = resolved(root, args.output)
    out.mkdir(parents=True, exist_ok=True)

    basin_path = root / f"raw_data/hybas_af_lev01-12_v1c/hybas_af_lev{level}_v1c.shp"
    river_dir = root / "raw_data/HydroRIVERS_v10_af"
    river_path = river_dir / "HydroRIVERS_v10_af.gdb"
    if not river_path.exists():
        matches = sorted(river_dir.rglob("*.gdb")) if river_dir.exists() else []
        river_path = matches[0] if matches else river_path
    sources_path = root / PROJECT_BASE / "06c_hydrological_spatial_mapping/01_hydrology_source_locations.csv"
    candidate_path = root / PROJECT_BASE / "06e_river_basin_connectivity/04_downstream_admin_candidates_REVIEW.csv"
    if not candidate_path.exists():
        candidate_path = root / PROJECT_BASE / "06f_river_basin_connectivity/04_downstream_admin_candidates_REVIEW.csv"

    adm2_path = root / "raw_data/administrative_boundaries/ssd_admin2.geojson"
    adm3_path = root / "raw_data/administrative_boundaries/ssd_admin3.geojson"
    adm2 = load_admin(adm2_path, "ADM2")
    adm3 = load_admin(adm3_path, "ADM3")
    print(f"Administrative units: ADM2={len(adm2)}, ADM3={len(adm3)}")
    basins, n_basins = load_basins(basin_path, tuple(adm2.total_bounds), pad_km=100)
    print(f"Nearby level {level} basins: {n_basins}")

    crosswalk = resolved(root, args.system_crosswalk) if args.system_crosswalk else None
    candidates = read_candidates(candidate_path)
    sources = read_sources(sources_path)
    # Visible major-river subset for map only, not used to assign hydrological systems.
    bounds = adm2.unary_union.buffer(150_000)
    rivers = load_rivers(river_path, bounds, args.stream_order_min)
    print(f"Major river segments to display: {len(rivers)}")

    report = {"basin_level": level, "boundary_sources": {
        "adm2": str(adm2_path), "adm3": str(adm3_path)},
        "basin_source": str(basin_path), "river_source": str(river_path),
        "system_crosswalk": str(crosswalk) if crosswalk else None,
        "hydrology_connections_are_unverified": True,
        "overlap_minimum_percent": args.min_overlap_percent,
        "levels": {}}
    combined = []
    for level_name, adm in [("ADM2", adm2), ("ADM3", adm3)]:
        overlaps = calculate_overlaps(adm, basins, level_name, args.min_overlap_percent)
        overlaps = apply_crosswalk(overlaps, crosswalk)
        main, by_system = summaries(overlaps)
        main = attach_candidate_summary(main, candidates)
        overlaps.to_csv(out / f"{level_name.lower()}_all_basin_overlaps.csv", index=False)
        main.to_csv(out / f"{level_name.lower()}_dominant_basin.csv", index=False)
        by_system.to_csv(out / f"{level_name.lower()}_system_area_shares.csv", index=False)
        report["levels"][level_name] = draw_map(adm, main, basins, rivers, sources, level_name,
            out / f"{level_name.lower()}_major_basins_map.png",
            out / f"{level_name.lower()}_dominant_basins.gpkg")
        combined.append(main)
        print(f"{level_name}: computed {len(overlaps)} basin overlaps; "
              f"dominant basin mapped to {report['levels'][level_name]['mapped_polygons']} ADM units.")

    pd.concat(combined, ignore_index=True).to_csv(out / "admin_dominant_basins_and_source_candidates.csv", index=False)
    all_ids = basins[["HYBAS_ID", "NEXT_DOWN"]].drop_duplicates().sort_values("HYBAS_ID")
    all_ids["hydrological_system"] = ""
    all_ids["evidence_reference"] = ""
    all_ids["review_status"] = "PENDING"
    all_ids.to_csv(out / "reviewed_system_crosswalk_TEMPLATE.csv", index=False)
    (out / "methodology.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Results:", out)
    print("NOTE: Named systems are UNCLASSIFIED without a reviewed crosswalk.")
    print("NOTE: Source-to-ADM links remain UNVERIFIED and are not model features.")


if __name__ == "__main__":
    main()
