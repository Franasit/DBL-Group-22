#!/usr/bin/env python3
"""08c — South Sudan flood-activity change × hydrological connectivity.

Input (already generated):
  processed_data/multivariate_flood_modeling/08b_before_after_2019/
      adm2_all_period_statistics.csv, adm3_all_period_statistics.csv
  processed_data/multivariate_flood_modeling/06e_river_basin_connectivity/
      04_downstream_admin_candidates_REVIEW.csv
  raw_data/administrative_boundaries/ssd_admin2.geojson, ssd_admin3.geojson

The outputs are descriptive historical overlays, NOT validated causal links.
No hydrology features are approved or sent to prediction models.

Default run: python scripts/multivariate_flood_modeling/08c_flood_trend_hydrological_connectivity.py
Dependencies: pandas, geopandas, matplotlib, numpy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

PALETTE = {
    "improved": "#7EC9F2",  # brighter sky blue
    "stable": "#F4F0F8",    # light neutral
    "worsened": "#C18AE0", # brighter violet
    "strong": "#7546B7",   # vivid but balanced purple
    "missing": "#E2E2E2",
}
BASE = "processed_data/multivariate_flood_modeling"
REQUIRED_STATS = ["adm_id", "region_name", "parent_adm2", "mean_days_per_year_2000_2018",
                  "mean_days_per_year_2019_2025", "mean_days_per_year_2000_2025",
                  "change_recent_minus_pre_days_per_year"]
CAND_KEYS = ["level", "adm_id", "source_type", "source_id", "downstream_hops"]


def project_root() -> Path:
    here = Path(__file__).resolve()
    if here.parent.name == "multivariate_flood_modeling" and here.parent.parent.name == "scripts":
        return here.parents[2]
    return Path.cwd().resolve()


def resolve_path(root, value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else root / p


def load_stats(path, level):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run 08b Top20 script first.")
    df = pd.read_csv(path, dtype={"adm_id": "string"})
    absent = [c for c in REQUIRED_STATS if c not in df.columns]
    if absent:
        raise ValueError(f"Wrong 08b file {path.name}; missing: {absent}")
    df["adm_id"] = df["adm_id"].str.strip()
    if df["adm_id"].duplicated().any():
        raise ValueError(f"Duplicate ADM IDs in {path}")
    for c in REQUIRED_STATS[3:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "eligible_change" in df.columns:
        eligible = df["eligible_change"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        ypre = pd.to_numeric(df.get("n_years_2000_2018", pd.Series(np.nan,index=df.index)),errors="coerce")
        ypost = pd.to_numeric(df.get("n_years_2019_2025", pd.Series(np.nan,index=df.index)),errors="coerce")
        eligible = ypre.ge(15) & ypost.ge(5)
    df["eligible_change"] = eligible
    df.loc[~eligible, "change_recent_minus_pre_days_per_year"] = np.nan
    df["level"] = level
    return df



def build_stats_from_existing_annual_panel(root, level):
    """Build 08c-compatible statistics from the existing annual panel.

    Does not modify 08b outputs. Used when only a legacy 08b summary exists
    without reliable ADM codes.
    """
    level_l = level.lower()
    main_file = root / f"processed_data/multivariate_flood_modeling/{'01_adm2_panel_dataset' if level == 'ADM2' else '02_adm3_panel_dataset'}/full_analysis_panel.csv"
    if not main_file.exists():
        # A narrow search, checking real columns rather than guessing by filename.
        candidates = list((root / "processed_data/multivariate_flood_modeling").rglob("full_analysis_panel.csv"))
        for candidate in candidates:
            columns = pd.read_csv(candidate, nrows=0).columns
            if "flood__flood_days" in columns and f"flood__{level}_NAME" in columns:
                main_file = candidate
                break
        else:
            raise FileNotFoundError(
                f"No 08b Top20 statistics or suitable {level} annual panel found. "
                f"Expected: {main_file}"
            )
    frame = pd.read_csv(main_file, low_memory=False)
    required = ["adm_id", "year", "flood__flood_days", f"flood__{level}_NAME"]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"{main_file} is missing {missing}")
    work = pd.DataFrame({
        "adm_id": frame["adm_id"].astype("string").str.strip(),
        "year": pd.to_numeric(frame["year"], errors="coerce"),
        "activity": pd.to_numeric(frame["flood__flood_days"], errors="coerce"),
        "region_name": frame[f"flood__{level}_NAME"].astype("string"),
        "parent_adm2": frame["flood__ADM2_NAME"].astype("string") if level == "ADM3"
                           else pd.Series(pd.NA, index=frame.index, dtype="string"),
    })
    if "target_row_present" in frame.columns:
        observed = frame["target_row_present"].astype(str).str.lower().isin(["true", "1", "yes"])
        work.loc[~observed, "activity"] = np.nan
    work = work[work["year"].between(2000, 2025)].copy()
    if work.duplicated(["adm_id", "year"]).any():
        raise ValueError(f"{level} annual panel contains duplicate area-year rows.")
    if ((work["activity"] < 0) | (work["activity"] > 366)).any():
        raise ValueError(f"{level} annual panel includes invalid flood activity-day values.")
    out = []
    for code, group in work.groupby("adm_id"):
        series = group.set_index("year")["activity"].reindex(range(2000, 2026))
        pre = series.loc[2000:2018]
        recent = series.loc[2019:2025]
        names = group["region_name"].dropna()
        parents = group["parent_adm2"].dropna()
        out.append({
            "adm_id": str(code),
            "region_name": str(names.iloc[-1]) if len(names) else str(code),
            "parent_adm2": str(parents.iloc[-1]) if len(parents) else "",
            "mean_days_per_year_2000_2018": pre.mean(),
            "mean_days_per_year_2019_2025": recent.mean(),
            "mean_days_per_year_2000_2025": series.mean(),
            "change_recent_minus_pre_days_per_year": recent.mean() - pre.mean(),
            "n_years_2000_2018": int(pre.notna().sum()),
            "n_years_2019_2025": int(recent.notna().sum()),
            "eligible_change": bool(pre.notna().sum() >= 15 and recent.notna().sum() >= 5),
        })
    print(f"{level}: built compatible change statistics from {main_file}")
    return load_stats_frame(pd.DataFrame(out), level)


def load_stats_frame(df, level):
    df = df.copy()
    missing = [c for c in REQUIRED_STATS if c not in df.columns]
    if missing:
        raise ValueError(f"{level} statistics missing columns {missing}")
    if df.columns.duplicated().any():
        raise ValueError(f"{level}: duplicated statistics columns: {df.columns[df.columns.duplicated()].tolist()}")
    df["adm_id"] = df["adm_id"].astype("string").str.strip()
    if df["adm_id"].duplicated().any():
        raise ValueError(f"{level}: duplicate ADM IDs")
    for c in REQUIRED_STATS[3:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "eligible_change" in df:
        eligible = df["eligible_change"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        pre = pd.to_numeric(df.get("n_years_2000_2018"), errors="coerce")
        recent = pd.to_numeric(df.get("n_years_2019_2025"), errors="coerce")
        eligible = pre.ge(15) & recent.ge(5)
    df["eligible_change"] = eligible
    df.loc[~eligible, "change_recent_minus_pre_days_per_year"] = np.nan
    df["level"] = level
    return df


def load_stats_compatible(path, level, root):
    """Read 08b statistics with stable ADM codes; support both old and fixed 08b outputs."""
    if path.exists():
        return load_stats_frame(pd.read_csv(path, dtype={"adm_id": "string"}), level)

    legacy = path.parent / ("01_adm2_period_summary.csv" if level == "ADM2"
                            else "02_adm3_period_summary.csv")
    if legacy.exists():
        old = pd.read_csv(legacy, dtype={"adm_id": "string", "region_id": "string"})
        # Fixed 08b outputs contain *both* adm_id and region_id.  Never rename
        # region_id -> adm_id if adm_id already exists: duplicate column names
        # make df["adm_id"] a DataFrame rather than a Series.
        stable_id = None
        for field in ("adm_id", "region_id"):
            if field not in old.columns:
                continue
            codes = old[field].astype("string").str.strip()
            if codes.notna().all() and codes.str.fullmatch(r"SS[0-9]+", na=False).all():
                if not codes.duplicated().any():
                    stable_id = field
                    break
        if stable_id is not None:
            out = old.copy()
            if stable_id == "region_id":
                out = out.drop(columns=["adm_id"], errors="ignore")
                out = out.rename(columns={"region_id": "adm_id"})
            else:
                # Keep actual ADM IDs, discard redundant legacy identifier.
                out = out.drop(columns=["region_id"], errors="ignore")

            # Rename only target columns that do not already exist.
            aliases = {
                "parent_adm2_name": "parent_adm2",
                "mean_annual_flood_days_2000_2018": "mean_days_per_year_2000_2018",
                "mean_2019-2025": "mean_days_per_year_2019_2025",
                "mean_annual_flood_days_2000_2025": "mean_days_per_year_2000_2025",
                "change_after_2019": "change_recent_minus_pre_days_per_year",
                "years_2019-2025": "n_years_2019_2025",
            }
            for old_name, new_name in aliases.items():
                if old_name in out.columns:
                    if new_name in out.columns:
                        out = out.drop(columns=[old_name])
                    else:
                        out = out.rename(columns={old_name: new_name})
            if "parent_adm2" not in out.columns:
                out["parent_adm2"] = ""
            if "n_years_2000_2018" not in out.columns:
                a = pd.to_numeric(old.get("years_2000-2010"), errors="coerce")
                b = pd.to_numeric(old.get("years_2011-2018"), errors="coerce")
                out["n_years_2000_2018"] = a + b
            duplicates = out.columns[out.columns.duplicated()].tolist()
            if duplicates:
                raise ValueError(f"{level}: duplicated columns in adapted 08b summary: {duplicates}")
            print(f"{level}: loading 08b summary with stable ADM IDs from {legacy}")
            return load_stats_frame(out, level)
        print(f"{level}: 08b summary has no unique stable ADM code; deriving safely from annual panel")
    return build_stats_from_existing_annual_panel(root, level)


def load_candidates(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run 06e connectivity script first.")
    df = pd.read_csv(path, dtype={"adm_id": "string", "source_id": "string"})
    missing = sorted(set(CAND_KEYS) - set(df.columns))
    if missing:
        raise ValueError(f"Candidate CSV missing {missing}")
    for c in ["level", "adm_id", "source_type", "source_id"]:
        df[c] = df[c].astype("string").str.strip()
    df["level"] = df["level"].str.upper()
    df["source_type"] = df["source_type"].str.lower()
    df["downstream_hops"] = pd.to_numeric(df["downstream_hops"],errors="coerce")
    if "hydrological_connection_verified" in df:
        # Import as explicit boolean from the source file only; do not infer approval.
        df["hydrological_connection_verified"] = df["hydrological_connection_verified"].astype(str).str.lower().isin(["true","1","yes"])
    else:
        df["hydrological_connection_verified"] = False
    return df


def candidates_per_admin(df, level):
    d = df[df["level"] == level].dropna(subset=["adm_id","source_type","source_id"]).copy()
    # Multiple basin paths from one source to one admin count as ONE source.
    d = d.sort_values("downstream_hops").drop_duplicates(["adm_id","source_type","source_id"])
    summaries = []
    for aid, group in d.groupby("adm_id"):
        by_lake = group.loc[group["source_type"].eq("lake"), "source_id"].tolist()
        by_dfo = group.loc[group["source_type"].eq("dfo"), "source_id"].tolist()
        summaries.append({
            "adm_id":aid,
            "candidate_lake_count":len(by_lake),
            "candidate_dfo_count":len(by_dfo),
            "candidate_total_sources":len(group),
            "candidate_lake_sources":"; ".join(sorted(by_lake)),
            "candidate_dfo_sources":"; ".join(sorted(by_dfo)),
            "minimum_basin_hops":group["downstream_hops"].min(),
            "verified_link_count_in_06e":int(group["hydrological_connection_verified"].sum()),
        })
    return pd.DataFrame(summaries, columns=[
        "adm_id", "candidate_lake_count", "candidate_dfo_count", "candidate_total_sources",
        "candidate_lake_sources", "candidate_dfo_sources", "minimum_basin_hops", "verified_link_count_in_06e"])


def load_geometry(path, level):
    if not path.exists():
        raise FileNotFoundError(f"Missing boundaries: {path}")
    g = gpd.read_file(path)
    names = (["adm2_pcode", "ADM2_PCODE", "admin2Pcod", "adm_id"] if level == "ADM2"
             else ["adm3_pcode", "ADM3_PCODE", "admin3Pcod", "adm_id"])
    col = next((c for c in names if c in g.columns), None)
    if col is None:
        raise ValueError(f"Cannot find {level} pcode in {path}. Available columns: {list(g.columns)}")
    g["adm_id"] = g[col].astype("string").str.strip()
    if g["adm_id"].duplicated().any():
        g = g[["adm_id", "geometry"]].dissolve(by="adm_id", as_index=False)
    return g[["adm_id", "geometry"]].copy().to_crs("EPSG:6933")


def merge_level(stats, candidates, geometry, level):
    counts = candidates_per_admin(candidates, level)
    merged = stats.merge(counts, on="adm_id", how="left",validate="1:1")
    for col in ["candidate_lake_count","candidate_dfo_count", "candidate_total_sources", "verified_link_count_in_06e"]:
        merged[col] = merged[col].fillna(0).astype(int)
    for col in ["candidate_lake_sources","candidate_dfo_sources"]:
        merged[col] = merged[col].fillna("")
    merged["network_candidate_present"] = merged["candidate_total_sources"].gt(0)
    merged["connectivity_status"] = np.where(merged["network_candidate_present"],
                                             "UNVERIFIED_NETWORK_CANDIDATE", "NO_NETWORK_CANDIDATE")
    joined = geometry.merge(merged, on="adm_id", how="left",validate="1:1")
    matched = int(joined["region_name"].notna().sum())
    print(f"{level}: spatially matched {matched}/{len(geometry)} regions; network candidate in {merged['network_candidate_present'].sum()} units")
    if not matched:
        raise ValueError(f"{level}: zero geometry-to-statistics matches; check ADM Pcodes.")
    return merged, joined


def select_top(df, kind, n=20):
    base=df[df["eligible_change"]].copy()
    c="change_recent_minus_pre_days_per_year"
    if kind == "highest":
        base=base.sort_values(["mean_days_per_year_2000_2025","adm_id"],ascending=[False,True])
    elif kind == "worsening":
        base=base[base[c]>0].sort_values([c,"adm_id"],ascending=[False,True])
    elif kind == "improving":
        base=base[base[c]<0].sort_values([c,"adm_id"],ascending=[True,True])
    else:
        raise ValueError(kind)
    base=base.head(n).copy()
    base.insert(0,"rank",range(1,len(base)+1))
    return base


def plot_level(gdf, top, level, output, kind="worsening"):
    """Poster layout: dedicated non-overlapping map, legend, colorbar, table, footer."""
    from matplotlib.colors import LinearSegmentedColormap
    vals = pd.to_numeric(gdf["change_recent_minus_pre_days_per_year"], errors="coerce")
    finite = vals[np.isfinite(vals)]
    lim = max(1.0, float(np.percentile(np.abs(finite), 95))) if len(finite) else 1.0
    cool = LinearSegmentedColormap.from_list("blue_purple", [
        (0.0, PALETTE["improved"]), (0.5, PALETTE["stable"]),
        (0.76, PALETTE["worsened"]), (1.0, PALETTE["strong"])
    ])
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    fig = plt.figure(figsize=(19, 10.5), facecolor="#FCFBFE")
    # Larger map panel; right table narrowed slightly to give the map more presence.
    ax = fig.add_axes([0.025, 0.135, 0.670, 0.770])
    tx = fig.add_axes([0.735, 0.170, 0.245, 0.710]); tx.axis("off")
    gdf.plot(ax=ax, color=PALETTE["missing"], edgecolor="#ECECEC", linewidth=.15)
    colored = gdf[gdf["eligible_change"].fillna(False) & vals.notna()].copy()
    if not colored.empty:
        colored.plot(ax=ax, column="change_recent_minus_pre_days_per_year",
                     cmap=cool, norm=norm, edgecolor="white", linewidth=.13)
    ax.set_axis_off()
    ax.set_title(f"{level} | Recorded flood-activity change (2019–2025 vs 2000–2018)",
                 fontsize=12.2, fontweight="bold", pad=9)
    if not top.empty and not colored.empty:
        markers = colored.merge(top[["adm_id", "rank"]], on="adm_id", how="inner")
        for _, r in markers.iterrows():
            pt = r.geometry.representative_point()
            ax.annotate(str(int(r["rank"])), (pt.x, pt.y), fontsize=7.5,
                        color="white", fontweight="bold", ha="center", va="center",
                        bbox={"boxstyle": "circle,pad=.16", "fc": "#353044", "ec": "white", "lw": .6}, zorder=10)

    # Legend in its own fixed figure-level space, never inside the map.
    handles = [
        Patch(facecolor=PALETTE["improved"], label="Decreased"),
        Patch(facecolor=PALETTE["stable"], label="Near zero"),
        Patch(facecolor=PALETTE["strong"], label="Increased"),
        Patch(facecolor=PALETTE["missing"], label="No data"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.360, 0.078),
               ncol=4, frameon=False, fontsize=9, columnspacing=1.2, handlelength=1.5)
    # Short vertical colorbar between map and table.
    if not colored.empty:
        cax = fig.add_axes([.625, .225, .012, .185])
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cool), cax=cax,
                          orientation="vertical")
        cb.ax.tick_params(labelsize=8, pad=2)
        cb.set_label("Δ mean activity\n(days/year)", fontsize=8.6, labelpad=6)

    cols = ["rank", "region_name"] + (["parent_adm2"] if level == "ADM3" else []) + [
        "change_recent_minus_pre_days_per_year", "candidate_total_sources"]
    display = top[cols].copy()
    if not display.empty:
        display["change_recent_minus_pre_days_per_year"] = display[
            "change_recent_minus_pre_days_per_year"].map(lambda z: f"{z:+.1f}")
        display.columns = (["#", "ADM3", "Parent ADM2", "Δ days/yr", "Candidates"]
                           if level == "ADM3" else ["#", "ADM2", "Δ days/yr", "Candidates"])
    tx.set_title(f"Top {len(top)} increasing areas", loc="left", fontsize=13.5,
                 fontweight="bold", pad=12)
    if not display.empty:
        widths = [.07, .32, .28, .18, .15] if level == "ADM3" else [.09, .45, .24, .22]
        tbl = tx.table(cellText=display.astype(str).values.tolist(),
                       colLabels=list(display.columns), cellLoc="left", colLoc="left",
                       colWidths=widths, bbox=[0, .085, 1, .845])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.1)
        for (ri, ci), cell in tbl.get_celld().items():
            cell.set_edgecolor("#DAD8E2")
            cell.set_linewidth(.4)
            if ri == 0:
                cell.set_facecolor("#E4DBEF")
                cell.set_text_props(fontweight="bold", color="#272536")
            elif ri % 2 == 0:
                cell.set_facecolor("#F7F5FA")

    fig.suptitle(f"South Sudan | Historical Flood Activity and Hydrological Network Candidates ({level})",
                 fontsize=17.5, fontweight="bold", x=.5, y=.962)
    fig.text(.030, .032,
             "Recorded activity dates ≠ full-area inundation duration or humanitarian impact. "
             "Lake/DFO links are UNVERIFIED network candidates.",
             fontsize=9, color="#494553", ha="left")
    fig.savefig(output, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_parent_summary(df, output):
    frame=df.copy()
    frame["parent_adm2"] = frame["parent_adm2"].fillna("Unknown")
    counts=frame.groupby("parent_adm2").agg(
        top20_adm3_count=("adm_id","nunique"),
        with_network_candidate=("network_candidate_present","sum"),
        mean_change_days_per_year=("change_recent_minus_pre_days_per_year","mean")
    ).reset_index().sort_values(["top20_adm3_count","parent_adm2"],ascending=[False,True])
    counts.to_csv(output,index=False)


def run_level(root, level, stats_path, geometry_path, candidates, out, top_n, top_map_n):
    stats=load_stats_compatible(stats_path,level,root)
    geom=load_geometry(geometry_path,level)
    combined, map_gdf=merge_level(stats,candidates,geom,level)
    combined.to_csv(out/f"{level.lower()}_all_change_connectivity.csv",index=False)
    for kind in ("highest","worsening","improving"):
        top=select_top(combined,kind,top_n)
        top.to_csv(out/f"{level.lower()}_top{top_n}_{kind}_connectivity.csv",index=False)
        if kind=="worsening":
            plot_level(map_gdf,top.head(top_map_n),level,out/f"{level.lower()}_worsening_map.png",kind)
        if level=="ADM3":
            plot_parent_summary(top,out/f"adm3_top{top_n}_{kind}_by_parent_adm2.csv")
    return {"regions":int(len(combined)),"with_candidate":int(combined["network_candidate_present"].sum()),
            "with_comparable_trend":int(combined["eligible_change"].sum()),
            "matched_geometry":int(map_gdf["region_name"].notna().sum())}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",default=str(project_root()))
    ap.add_argument("--stats-dir",default=f"{BASE}/08b_before_after_2019")
    ap.add_argument("--connections",default=f"{BASE}/06e_river_basin_connectivity/04_downstream_admin_candidates_REVIEW.csv")
    ap.add_argument("--adm2-geometry",default="raw_data/administrative_boundaries/ssd_admin2.geojson")
    ap.add_argument("--adm3-geometry",default="raw_data/administrative_boundaries/ssd_admin3.geojson")
    ap.add_argument("--output",default=f"{BASE}/08c_flood_trend_hydrological_connectivity")
    ap.add_argument("--top-n",type=int,default=20)
    ap.add_argument("--map-numbered",type=int,default=12)
    args=ap.parse_args()
    root=resolve_path(Path.cwd(),args.root).resolve()
    statsdir=resolve_path(root,args.stats_dir)
    out=resolve_path(root,args.output);out.mkdir(parents=True,exist_ok=True)
    cp=resolve_path(root,args.connections)
    if not cp.exists():
        alternate=resolve_path(root,f"{BASE}/06f_river_basin_connectivity/04_downstream_admin_candidates_REVIEW.csv")
        if alternate.exists(): cp=alternate
    conn=load_candidates(cp)
    results={}
    for level in ["ADM2","ADM3"]:
        stat=statsdir/f"{level.lower()}_all_period_statistics.csv"
        geo=resolve_path(root,getattr(args,f"{level.lower()}_geometry"))
        results[level]=run_level(root,level,stat,geo,conn,out,args.top_n,args.map_numbered)
    method={
        "analysis":"Descriptive recorded flood-activity change × candidate hydrological network connectivity",
        "periods":{"before":"2000–2018","after":"2019–2025"},
        "change":"Mean annual recorded flood-activity days (after minus before)",
        "ranking":"Highest long-term mean; largest positive change; largest negative change",
        "input_statistics":str(statsdir),"input_candidates":str(cp),
        "candidate_count_interpretation":"Number of distinct lake/DFO sources per ADM, NOT number of network paths or confirmed connections",
        "never_approved_automatically":True,
        "limits":[
            "Candidate downstream basin paths require review; do not infer causation or flood-wave travel time.",
            "Do not use the 2000–2025 trend classification as a predictive training feature for held-out 2023–2025 tests.",
            "Remote-sensing observation completeness, permanent water exclusion, and spatial detection limitations are not independently verified.",
            "Changes in recorded activity do not independently establish changes in humanitarian disaster severity.",
        ],"results":results}
    (out/"methodology.json").write_text(json.dumps(method,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"output":str(out),"results":results,"note":"All connections remain UNVERIFIED candidates"},indent=2,ensure_ascii=False))

if __name__=="__main__":
    main()
