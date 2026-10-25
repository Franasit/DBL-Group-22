#!/usr/bin/env python3
"""
13b_integrate_adm3_impact_data.py
JBG060 — South Sudan ADM3 Impact Data Integration

Read-only input processing. No arbitrary priority score is calculated.

INPUTS
 raw_data/administrative_boundaries/ssd_admin3.geojson
 raw_data/worldpop/ssd_pop_YYYY_CN_100m_*.tif
 raw_data/health_facilities/*.geojson (or .gpkg/.shp)
 raw_data/farmland/asap_mask_crop_v04.tif
 raw_data/farmland/asap_mask_rangeland_v04.tif
 raw_data/farmland/geonode__cattle_gha.tif (optional)
 raw_data/IPC/*.xlsx (optional; review-only, not automatically matched)

OUTPUTS
 processed_data/multivariate_flood_modeling/13b_adm3_impact_integration/
   01_adm3_impact_indicators.csv
   02_worldpop_adm3_yearly.csv
   03_health_facility_counts.csv
   04_agricultural_land_area.csv
   05_ipc_review_inventory.csv
   06_data_quality_flags.csv
   07_methodology.json
   08_adm3_impact_coverage_map.png / .pdf

IMPORTANT DEFINITIONS
 WorldPop "CN" population-count rasters: raster sums are population estimates,
 NOT observed flood-exposed people. Confirm units against source metadata.
 ASAP crop / rangeland: percentage (0–100) per cell. Convert to km² using
 actual WGS84 cell ground area; grid cells are selected by centre.
 Cattle: older regional estimate; disabled by default. Summing its raster
 requires confirmation that each pixel is a COUNT, not a density.
 IPC is read for metadata and matching candidates ONLY. No guessed IPC score,
 interpolation, assignment to unmatched ADM3, or use of forecast IPC releases.

Packages: geopandas, rasterio, pandas, numpy, matplotlib, pyproj, openpyxl
Run from the project root:
  python scripts/multivariate_flood_modeling/13b_integrate_adm3_impact_data.py

Options:
  --population-years 2023 2024 2025
  --include-cattle
  --skip-population
  --skip-agriculture
  --skip-health
  --skip-ipc
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
from rasterio.mask import mask as raster_mask
from pyproj import Geod
from shapely.geometry import mapping

GEOD = Geod(ellps="WGS84")
CRS = "EPSG:4326"
PALETTE = {
    "complete": "#7546B7", "partial": "#C18AE0",
    "missing": "#E5E1E9", "border": "#48414E",
}

def project_root():
    p = Path(__file__).resolve()
    if p.parent.name == "multivariate_flood_modeling" and p.parent.parent.name == "scripts":
        return p.parents[2]
    return Path.cwd().resolve()

def first_existing(columns, candidates):
    lookup = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None

def normalize_id(x):
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    if not s or s in {"NAN", "NONE", "NULL"}:
        return None
    return s

def boundaries(root):
    file = root / "raw_data/administrative_boundaries/ssd_admin3.geojson"
    if not file.exists():
        raise FileNotFoundError(f"ADM3 boundary missing: {file}")
    g = gpd.read_file(file)
    if g.crs is None:
        raise ValueError("ADM3 boundary has no CRS; cannot reliably geolocate data.")
    g = g.to_crs(CRS)
    col = first_existing(g.columns, ["adm_id", "ADM3_PCODE", "ADM3_CODE", "ADM3_ID", "pcode"])
    if col is None:
        raise ValueError("No ADM3 code found in administrative boundary columns: " + str(list(g.columns)))
    a3 = first_existing(g.columns, ["ADM3_NAME", "adm3_name", "ADM3_EN", "name_3", "admin3Name"])
    a2 = first_existing(g.columns, ["ADM2_NAME", "adm2_name", "ADM2_EN", "name_2", "admin2Name"])
    out = g[["geometry"]].copy()
    out["adm_id"] = g[col].map(normalize_id)
    out["adm3_name"] = g[a3].astype(str) if a3 else out["adm_id"]
    out["adm2_name"] = g[a2].astype(str) if a2 else ""
    if out["adm_id"].isna().any() or out["adm_id"].duplicated().any():
        raise ValueError("Missing or duplicate ADM3 IDs in boundaries; review before aggregating.")
    if out.geometry.isna().any() or out.geometry.is_empty.any():
        raise ValueError("Some ADM3 geometries are missing or empty.")
    return out

def pixel_area_km2(src, row_indices):
    """Area of one pixel per row (WGS84 north-up raster), in km2."""
    a = src.transform
    if src.crs is None or src.crs.to_epsg() != 4326:
        raise ValueError("ASAP area integration expects EPSG:4326 lat/lon raster.")
    if abs(a.b) > 1e-10 or abs(a.d) > 1e-10:
        raise ValueError("Rotated raster grid: pixel area algorithm not applicable.")
    left = a.c
    right = left + a.a
    unique_rows = np.unique(row_indices)
    result = {}
    for row in unique_rows:
        lat_top = a.f + row * a.e
        lat_bottom = lat_top + a.e
        lon = [left, right, right, left]
        lat = [lat_top, lat_top, lat_bottom, lat_bottom]
        area, _ = GEOD.polygon_area_perimeter(lon, lat)
        result[int(row)] = abs(area) / 1e6
    return result

def zonal_raster(g, raster_path, mode):
    """
    Region-by-region raster extraction with masked crop.
    Returns value, number of valid cells, sampled cells, covered fraction.
    At country borders: cell-centre inclusion approximation.
    """
    rows = []
    with rasterio.open(raster_path) as ds:
        if ds.crs is None:
            raise ValueError("Raster CRS undefined: " + str(raster_path))
        reproj = g.to_crs(ds.crs)
        if ds.count < 1:
            raise ValueError("Raster has no bands: " + str(raster_path))
        for idx, rec in enumerate(reproj.itertuples(index=False)):
            adm_id = rec.adm_id
            try:
                cropped, tfm = raster_mask(
                    ds, [mapping(rec.geometry)], crop=True,
                    filled=False, all_touched=False, indexes=1
                )
            except ValueError:  # No overlap with raster
                rows.append({"adm_id": adm_id, "value": np.nan,
                             "valid_cells": 0, "sampled_cells": 0, "coverage_frac": 0.0})
                continue
            arr = np.ma.asarray(cropped)
            valid = ~np.ma.getmaskarray(arr) & np.isfinite(np.ma.getdata(arr))
            data = np.ma.getdata(arr)
            valid &= data >= 0
            valid_n = int(valid.sum())
            # A masked crop has NoData outside polygon, so this is a *pixel-mask*
            # count, not verified satellite observation coverage.
            geom_mask = rasterio.features.geometry_mask(
                [mapping(rec.geometry)], out_shape=data.shape,
                transform=tfm, invert=True, all_touched=False
            )
            eligible = int(geom_mask.sum())
            coverage = valid_n / eligible if eligible else 0.0
            val = np.nan
            if valid_n:
                samples = data[valid].astype(np.float64)
                if mode == "count":
                    val = float(samples.sum())
                elif mode == "fraction_area":
                    if np.any(samples > 100.00001):
                        raise ValueError(f"{raster_path.name}: land fraction >100 detected for {adm_id}; inspect scale.")
                    ri, _ = np.nonzero(valid)
                    # Transform is crop-local, but original raster transform must
                    # be used for an equivalent global row coordinate.
                    global_start = int(round((tfm.f - ds.transform.f) / ds.transform.e))
                    areas = pixel_area_km2(ds, ri + global_start)
                    weights = np.fromiter((areas[int(r)] for r in ri + global_start),
                                          dtype=float, count=len(ri))
                    val = float(np.sum(weights * samples / 100.0))
                else:
                    raise ValueError(mode)
            rows.append({"adm_id": adm_id, "value": val,
                         "valid_cells": valid_n, "sampled_cells": eligible,
                         "coverage_frac": coverage})
            if (idx + 1) % 100 == 0:
                print(f"  {raster_path.name}: {idx+1}/{len(g)} ADM3", flush=True)
    return pd.DataFrame(rows)

def worldpop(g, root, years):
    folder = root / "raw_data/worldpop"
    if not folder.exists():
        warnings.warn("WorldPop folder missing; columns remain unavailable")
        return pd.DataFrame(columns=["adm_id","year","population_estimate","population_valid_pixels","population_data_fraction"])
    results = []
    for year in years:
        hits = sorted(folder.glob(f"ssd_pop_{year}_CN_100m_*.tif"))
        if not hits:
            warnings.warn(f"No WorldPop count raster found for {year}")
            continue
        if len(hits) > 1:
            raise ValueError(f"Ambiguous WorldPop rasters for {year}: {hits}")
        print(f"WorldPop {year}: {hits[0].name}", flush=True)
        z = zonal_raster(g, hits[0], "count")
        z = z.rename(columns={"value":"population_estimate",
                              "valid_cells":"population_valid_pixels",
                              "coverage_frac":"population_data_fraction"})
        z["year"] = year
        results.append(z[["adm_id","year","population_estimate",
                          "population_valid_pixels","population_data_fraction"]])
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(
        columns=["adm_id","year","population_estimate","population_valid_pixels","population_data_fraction"])

def agriculture(g, root, include_cattle):
    folder = root / "raw_data/farmland"
    mapping_files = {
        "cropland_area_km2": ("asap_mask_crop_v04.tif", "fraction_area"),
        "rangeland_area_km2": ("asap_mask_rangeland_v04.tif", "fraction_area")
    }
    if include_cattle:
        mapping_files["cattle_count_2010_estimate_UNVERIFIED"] = (
            "geonode__cattle_gha.tif", "count"
        )
    result = g[["adm_id"]].copy()
    for output, (filename, mode) in mapping_files.items():
        path = folder / filename
        if not path.exists():
            warnings.warn(f"Missing agriculture raster: {path}")
            result[output] = np.nan
            result[f"{output}_data_fraction"] = np.nan
            continue
        print(f"Agriculture: {filename}", flush=True)
        z = zonal_raster(g, path, mode)
        z = z.rename(columns={"value":output,
                              "coverage_frac":f"{output}_data_fraction",
                              "valid_cells":f"{output}_valid_cells"})
        result = result.merge(z[["adm_id",output,f"{output}_data_fraction",
                                 f"{output}_valid_cells"]], on="adm_id", validate="one_to_one")
    return pd.DataFrame(result.drop(columns="geometry",errors="ignore"))

def facilities(g, root):
    """Count only valid point facilities; save a transparent geometry/coverage audit."""
    folder = root / "raw_data/health_facilities"
    files = ([p for ext in ("*.geojson", "*.gpkg", "*.shp")
              for p in folder.rglob(ext)] if folder.exists() else [])
    if not files:
        warnings.warn("No health facility vector file found.")
        empty = pd.DataFrame({"adm_id":g["adm_id"],
                              "health_facilities_total":np.nan})
        audit = pd.DataFrame([{"category":"FILE_MISSING","n":0}])
        return empty, pd.DataFrame(), audit
    file = sorted(files)[0]
    print("Health:", file.name, flush=True)
    facilities = gpd.read_file(file)
    if facilities.crs is None:
        raise ValueError(f"Facility geometry CRS missing: {file}")
    original_n = len(facilities)
    country = first_existing(facilities.columns,
                             ["Country","country","COUNTRY","Country_Name","admin0"])
    if country:
        country_values = facilities[country].astype(str).str.strip().str.lower()
        facilities = facilities[country_values.isin(["south sudan","ssd","ss"])].copy()
    else:
        warnings.warn("No country field: facilities will be spatially limited to the ADM3 boundary.")
    selected_n = len(facilities)
    # GeoJSON often includes missing geometries (None/NaN); these are NOT
    # automatically invalid healthcare facilities, just unlocatable records.
    missing = facilities.geometry.isna()
    nonempty = ~missing & ~facilities.geometry.is_empty
    types_before = facilities.geometry.geom_type.fillna("Missing").value_counts()
    point_mask = nonempty & facilities.geometry.geom_type.isin(["Point", "MultiPoint"])
    usable = facilities.loc[point_mask].copy()
    rejected = selected_n - len(usable)
    # MultiPoints should not be silently counted as one individual facility.
    # Explode to point geometries, retaining source row ID for ambiguity audit.
    usable["_source_record"] = usable.index.astype(str)
    usable = usable.to_crs(g.crs)
    if len(usable):
        usable = usable.explode(index_parts=False).reset_index(drop=True)
        usable["_point_id"] = np.arange(len(usable))
        located = gpd.sjoin(usable, g[["adm_id", "geometry"]],
                            how="inner", predicate="intersects")
        hits = located.groupby("_point_id")["adm_id"].nunique()
        ambiguous_ids = set(hits[hits > 1].index)
        ambiguous_records = located[located["_point_id"].isin(ambiguous_ids)].copy()
        assigned = located[~located["_point_id"].isin(ambiguous_ids)].copy()
        # any duplicate points because of multiple polygons in same ADM is
        # safely removed before counting.
        assigned = assigned.drop_duplicates(subset=["_point_id", "adm_id"])
        unassigned_points = len(usable) - len(hits)
    else:
        assigned = gpd.GeoDataFrame(columns=["adm_id"], geometry=gpd.GeoSeries([], crs=g.crs))
        ambiguous_records = assigned.copy()
        unassigned_points = 0
        ambiguous_ids = set()
    counts = assigned.groupby("adm_id").size().rename("health_facilities_total")
    out = pd.DataFrame({"adm_id":g["adm_id"]}).merge(
        counts, left_on="adm_id", right_index=True, how="left")
    out["health_facilities_total"] = out["health_facilities_total"].fillna(0).astype(int)
    # Audit counts are global, not fake ADM3-level flags.
    audit_rows = [
        {"category":"all_input_records", "n":original_n},
        {"category":"south_sudan_attribute_selected", "n":selected_n},
        {"category":"valid_point_or_multipoint_records", "n":len(facilities.loc[point_mask])},
        {"category":"excluded_missing_empty_or_nonpoint_records", "n":rejected},
        {"category":"usable_point_geometries_after_explode", "n":len(usable)},
        {"category":"point_geometries_outside_adm3_polygons", "n":unassigned_points},
        {"category":"point_geometries_on_multiple_adm3_boundaries_excluded", "n":len(ambiguous_ids)},
        {"category":"point_geometries_assigned_to_one_adm3", "n":int(counts.sum())},
    ]
    audit_rows += [{"category":f"original_geometry_type_{t}", "n":int(n)}
                   for t,n in types_before.items()]
    audit = pd.DataFrame(audit_rows)
    print("Health point geometry audit:", audit.to_dict("records"), flush=True)
    type_col = first_existing(assigned.columns,
                              ["Facility_t","facility_type","type","Facility Type"])
    if type_col and len(assigned):
        assigned["facility_type_audit"] = (assigned[type_col].fillna("Unknown")
                                          .astype(str).str.strip())
        types = (assigned.groupby(["adm_id","facility_type_audit"]).size()
                 .unstack(fill_value=0).reset_index())
    else:
        types = pd.DataFrame()
    return out, types, audit


def ipc_inventory(root, g):
    folder = root / "raw_data/IPC"
    records = []
    if not folder.exists():
        return pd.DataFrame([{"status":"NO_IPC_FOLDER"}])
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".xlsx",".xls",".csv"}:
            continue
        try:
            if path.suffix.lower() == ".csv":
                sheet_names = ["CSV"]
                previews = [pd.read_csv(path, nrows=8)]
            else:
                workbook = pd.ExcelFile(path)
                sheet_names = workbook.sheet_names
                previews = [pd.read_excel(workbook, sheet_name=s, nrows=8) for s in sheet_names]
            for sheet, preview in zip(sheet_names, previews):
                namecols = [str(c) for c in preview.columns]
                records.append({
                    "file":str(path.relative_to(root)), "sheet":sheet,
                    "columns_preview":" | ".join(namecols[:35]),
                    "sample_rows":len(preview),
                    "status":"REVIEW_REQUIRED",
                    "reason":"ADM level, dates, current vs projected phase and official area crosswalk must be verified before assigning IPC to ADM3"
                })
        except Exception as exc:
            records.append({"file":str(path.relative_to(root)), "sheet":"",
                            "status":"READ_ERROR", "reason":str(exc)[:400]})
    return pd.DataFrame(records if records else [{"status":"NO_IPC_SPREADSHEETS"}])

def draw_coverage(g, result, out):
    plot = g.merge(result.drop(columns=["adm3_name","adm2_name"],errors="ignore"),
                   on="adm_id", how="left", validate="one_to_one")
    fields = ["population_estimate_2025", "health_facilities_total",
              "cropland_area_km2", "rangeland_area_km2"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.subplots_adjust(left=.04,right=.98,top=.90,bottom=.13,wspace=.08,hspace=.12)
    title_map = {
        "population_estimate_2025":"Population (2025) — table availability",
        "health_facilities_total":"Health facility count — table availability",
        "cropland_area_km2":"Cropland estimated area — table availability",
        "rangeland_area_km2":"Rangeland estimated area — table availability"
    }
    for ax, field in zip(axes.flat, fields):
        mask = plot[field].notna() if field in plot else pd.Series(False,index=plot.index)
        if (~mask).any():
            plot.loc[~mask].plot(ax=ax,color=PALETTE["missing"],edgecolor="white",linewidth=.12)
        if mask.any():
            plot.loc[mask].plot(ax=ax,color=PALETTE["complete"],edgecolor="white",linewidth=.12)
        plot.boundary.plot(ax=ax,color=PALETTE["border"],linewidth=.10,alpha=.45)
        ax.set_title(f"{title_map[field]}\n{int(mask.sum())}/{len(plot)} ADM3 with a value",
                     fontsize=11, weight="bold")
        ax.set_axis_off()
        ax.set_aspect("equal")
    fig.suptitle("South Sudan — ADM3 Impact Data Availability",fontsize=18,weight="bold",y=.97)
    handles=[mpatches.Patch(facecolor=PALETTE["complete"],label="Computed numeric value (not validated exposure)"),
             mpatches.Patch(facecolor=PALETTE["missing"],label="Missing / not processed")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.065),ncol=2,frameon=False)
    fig.text(.5,.02,"Availability only. Valid value does not establish satellite observation completeness or actual flood exposure.",
             ha="center",fontsize=9,color="#544F5A")
    fig.savefig(out/"08_adm3_impact_coverage_map.png",dpi=220,bbox_inches="tight")
    fig.savefig(out/"08_adm3_impact_coverage_map.pdf",bbox_inches="tight")
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,default=project_root())
    ap.add_argument("--population-years",nargs="+",type=int,default=[2023,2024,2025])
    ap.add_argument("--include-cattle",action="store_true",
                    help="Experimental 2010 cattle count, only after manual source-unit verification")
    ap.add_argument("--reuse-population", action="store_true",
                    help="Reuse existing 02_worldpop_adm3_yearly.csv after checking its ADM3/year coverage; saves raster processing time.")
    ap.add_argument("--skip-population",action="store_true")
    ap.add_argument("--skip-health",action="store_true")
    ap.add_argument("--skip-agriculture",action="store_true")
    ap.add_argument("--skip-ipc",action="store_true")
    args=ap.parse_args()
    root=args.root.resolve()
    output=root/"processed_data/multivariate_flood_modeling/13b_adm3_impact_integration"
    output.mkdir(parents=True,exist_ok=True)
    g=boundaries(root)
    ids=g[["adm_id","adm3_name","adm2_name"]].copy()
    print("ADM3 boundaries:",len(g),flush=True)
    all_indicators=ids.copy()

    saved_pop = output / "02_worldpop_adm3_yearly.csv"
    if not args.skip_population and args.reuse_population:
        if not saved_pop.exists():
            raise FileNotFoundError(f"--reuse-population requested but file missing: {saved_pop}")
        pop = pd.read_csv(saved_pop, dtype={"adm_id":"string"})
        pop["adm_id"] = pop["adm_id"].map(normalize_id)
        required = {"adm_id","year","population_estimate","population_data_fraction"}
        if not required.issubset(pop):
            raise ValueError(f"Existing population CSV missing columns: {required-set(pop.columns)}")
        if pop.duplicated(["adm_id","year"]).any():
            raise ValueError("Existing population CSV has duplicated ADM3/year rows.")
        expected = pd.MultiIndex.from_product([set(g.adm_id),args.population_years],
                                               names=["adm_id","year"])
        found = pd.MultiIndex.from_frame(pop[["adm_id","year"]])
        if len(expected.difference(found)) or len(found.difference(expected)):
            raise ValueError("Population CSV does not match 512 ADM3 and requested years; rerun without --reuse-population.")
        print(f"Reusing completed WorldPop table: {saved_pop}", flush=True)
    elif not args.skip_population:
        pop=worldpop(g,root,args.population_years)
    else:
        pop=pd.DataFrame(columns=["adm_id","year","population_estimate",
                                  "population_valid_pixels","population_data_fraction"])
    pop.to_csv(output/"02_worldpop_adm3_yearly.csv",index=False)
    for year in args.population_years:
        q=pop.loc[pop.year.eq(year),["adm_id","population_estimate",
                                   "population_data_fraction"]].rename(
            columns={"population_estimate":f"population_estimate_{year}",
                     "population_data_fraction":f"population_data_fraction_{year}"})
        all_indicators=all_indicators.merge(q,on="adm_id",how="left",validate="one_to_one")

    if not args.skip_health:
        health,types,health_audit=facilities(g,root)
    else:
        health=ids[["adm_id"]].assign(health_facilities_total=np.nan)
        types=pd.DataFrame()
        health_audit=pd.DataFrame([{"category":"HEALTH_SKIPPED","n":0}])
    health_audit.to_csv(output/"03c_health_geometry_audit.csv",index=False)
    health.to_csv(output/"03_health_facility_counts.csv",index=False)
    if not types.empty:
        types.to_csv(output/"03b_health_facility_type_audit.csv",index=False)
    all_indicators=all_indicators.merge(health,on="adm_id",how="left",validate="one_to_one")

    if not args.skip_agriculture:
        ag=agriculture(g,root,args.include_cattle)
    else:
        ag=ids[["adm_id"]].assign(cropland_area_km2=np.nan,rangeland_area_km2=np.nan)
    ag.to_csv(output/"04_agricultural_land_area.csv",index=False)
    all_indicators=all_indicators.merge(ag,on="adm_id",how="left",validate="one_to_one")

    ipc=ipc_inventory(root,g) if not args.skip_ipc else pd.DataFrame([{"status":"SKIPPED"}])
    ipc.to_csv(output/"05_ipc_review_inventory.csv",index=False)
    all_indicators.to_csv(output/"01_adm3_impact_indicators.csv",index=False)

    stat_fields=[c for c in all_indicators if c.startswith(("population_estimate_","cropland_area_km2",
                                                       "rangeland_area_km2","health_facilities_total"))]
    flags=all_indicators[["adm_id","adm3_name"]].copy()
    for col in stat_fields:
        flags[f"{col}_missing"]=all_indicators[col].isna()
    for year in args.population_years:
        c=f"population_data_fraction_{year}"
        if c in all_indicators:
            flags[f"population_{year}_incomplete_pixel_coverage"]=all_indicators[c].lt(.99)
    for c in ("cropland_area_km2_data_fraction","rangeland_area_km2_data_fraction"):
        if c in all_indicators:
            flags[f"{c}_below_99pct"]=all_indicators[c].lt(.99)
    flags.to_csv(output/"06_data_quality_flags.csv",index=False)

    metadata={
      "purpose":"ADM3-level administrative screening of population, services and livelihoods; NOT flood-impact estimates",
      "boundary_count":len(g),
      "population_years":args.population_years,
      "population_units":"WorldPop count per raster pixel; verify input metadata",
      "raster_assignment":"cell centre inclusion, no fractional boundary pixel weighting",
      "asap_land_fraction":"0-100 percent multiplied by WGS84 geodesic pixel area",
      "raster_coverage_fraction":"non-NoData fraction in included polygon-centre pixels; NOT temporal observation coverage",
      "health":"Point-in-polygon count, excludes ambiguous multi-polygon boundary points",
      "ipc":"Review-only; no assignment to ADM3 pending crosswalk, periods and phase definition",
      "gdp":"Not joined: World Bank national indicators do not distinguish ADM3",
      "cattle":"Not used by default: old 2010 estimate and grid-unit verification required",
      "excluded_claims":[
         "ADM3 total population equals observed flood-exposed population",
         "Facility location inside a high-hazard ADM3 means facility is flooded",
         "Cropland/rangeland ADM3 total equals flood-damaged area",
         "Any rank or priority score is validated humanitarian need"
      ],
    }
    (output/"07_methodology.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    draw_coverage(g,all_indicators,output)
    print("\nSaved outputs:", output)
    print("ADM3 rows:",len(all_indicators))
    print("Null columns:\n", all_indicators.isna().sum().to_string())
    print("\nIMPORTANT: Impact indicators are administrative screening data, not predicted affected counts.")

if __name__=="__main__":
    main()
