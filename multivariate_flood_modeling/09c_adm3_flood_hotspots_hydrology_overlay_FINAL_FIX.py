#!/usr/bin/env python3
"""09c – South Sudan ADM3 flood-activity maps with hydrology context.

Generate two publication-style maps:
 A. Mean annual days with recorded flood activity, 2000–2025.
 B. Change in mean annual recorded activity: 2019–2025 minus 2000–2018.

For each map: top 12 numbered ADM3 areas, right-hand ADM3/parent ADM2
rankings, HydroRIVERS major rivers, DFO discharge stations, lake-level
observation points, national/ADM2 boundaries, legend and color scale.

These are historical descriptions, NOT flood forecasts, and daily observation
coverage/permanent-water exclusions are not independently validated.
Network symbols show positions only and DO NOT verify source-to-ADM connectivity.

Run from project root or VS Code without command-line arguments.
Dependencies: geopandas, pandas, numpy, matplotlib, shapely; pyogrio recommended.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import warnings

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

BASE = "processed_data/multivariate_flood_modeling"
BLUE, LILAC, PURPLE = "#7EC9F2", "#C18AE0", "#7546B7"
BG, INK, LIGHT = "#FCFBFE", "#302B38", "#F4F0F8"
DFO, LAKE, RIVER = "#E17B39", "#7546B7", "#287EB8"
AREA_CRS = "EPSG:6933"


def root_default():
    file = Path(__file__).resolve()
    return file.parents[2] if file.parent.name == "multivariate_flood_modeling" else Path.cwd().resolve()


def locate_annual(root: Path):
    first = root / f"{BASE}/02_adm3_panel_dataset/full_analysis_panel.csv"
    if first.exists():
        return first
    for p in (root / "processed_data").rglob("full_analysis_panel.csv"):
        try:
            cols = set(pd.read_csv(p, nrows=0).columns)
            if {"adm_id", "year", "flood__flood_days", "flood__ADM3_NAME"}.issubset(cols):
                return p
        except (OSError, ValueError):
            pass
    raise FileNotFoundError("ADM3 annual full_analysis_panel.csv missing; run Script 02 first.")


def annual_metrics(path: Path, pre_min_years: int = 15, post_min_years: int = 5):
    df = pd.read_csv(path, low_memory=False)
    needed = {"adm_id", "year", "flood__flood_days", "flood__ADM3_NAME", "flood__ADM2_NAME"}
    if not needed.issubset(df):
        raise ValueError(f"Annual panel missing columns {sorted(needed-set(df.columns))}")
    d = pd.DataFrame({
        "adm3_id": df["adm_id"].astype("string").str.strip(),
        "year": pd.to_numeric(df["year"], errors="coerce"),
        "days": pd.to_numeric(df["flood__flood_days"], errors="coerce"),
        "adm3_name": df["flood__ADM3_NAME"].astype("string"),
        "parent_adm2": df["flood__ADM2_NAME"].astype("string"),
    })
    if "target_row_present" in df:
        present = df["target_row_present"].astype(str).str.lower().isin({"true", "1", "yes"})
        d.loc[~present, "days"] = np.nan
    d = d[d.year.between(2000, 2025)].dropna(subset=["adm3_id", "year"])
    if d.duplicated(["adm3_id", "year"]).any():
        raise ValueError("Duplicate ADM3/year entries in annual panel")
    if ((d.days < 0) | (d.days > 366)).any():
        raise ValueError("Annual recorded flood-activity days outside 0–366")
    rows = []
    for aid, group in d.groupby("adm3_id"):
        vals = group.set_index("year")["days"].reindex(range(2000, 2026))
        pre, post = vals.loc[2000:2018], vals.loc[2019:2025]
        names = group.sort_values("year")["adm3_name"].dropna()
        parents = group.sort_values("year")["parent_adm2"].dropna()
        eligible_high = int(vals.notna().sum()) >= 21
        eligible_change = int(pre.notna().sum()) >= pre_min_years and int(post.notna().sum()) >= post_min_years
        rows.append({
            "adm3_id": str(aid),
            "adm3_name": str(names.iloc[-1]) if not names.empty else str(aid),
            "parent_adm2": str(parents.iloc[-1]) if not parents.empty else "Unknown",
            "mean_days_per_year_2000_2025": vals.mean(),
            "mean_days_per_year_2000_2018": pre.mean(),
            "mean_days_per_year_2019_2025": post.mean(),
            "change_days_per_year": post.mean()-pre.mean(),
            "valid_years": int(vals.notna().sum()),
            "valid_pre_years": int(pre.notna().sum()),
            "valid_post_years": int(post.notna().sum()),
            "eligible_hotspot": eligible_high,
            "eligible_change": eligible_change,
        })
    return pd.DataFrame(rows)


def boundary(root, level):
    p = root / f"raw_data/administrative_boundaries/ssd_admin{level[-1]}.geojson"
    if not p.exists():
        raise FileNotFoundError(p)
    g = gpd.read_file(p)
    if g.crs is None:
        raise ValueError(f"Boundary CRS missing: {p}")
    columns = ({"ADM3": ["adm3_pcode", "ADM3_PCODE", "admin3Pcod", "adm_id"],
                "ADM2": ["adm2_pcode", "ADM2_PCODE", "admin2Pcod", "adm_id"]})[level]
    field = next((c for c in columns if c in g), None)
    if field is None:
        raise ValueError(f"Cannot find {level} code in {list(g.columns)}")
    key = "adm3_id" if level == "ADM3" else "adm2_id"
    g[key] = g[field].astype("string").str.strip()
    # Keep only the join key and geometry. Boundary attributes may contain
    # adm3_name / parent_adm2, which would be suffixed to _x and _y after merge.
    # Descriptive names are taken from the validated annual panel instead.
    g = g[[key, "geometry"]].copy()
    if g[key].isna().any() or g[key].duplicated().any():
        raise ValueError(f"Missing or duplicate {level} IDs in boundary file: {p}")
    return g.to_crs(AREA_CRS)


def stations(root):
    p = root / f"{BASE}/06c_hydrological_spatial_mapping/01_hydrology_source_locations.csv"
    if not p.exists():
        warnings.warn("Station locations CSV not found; symbols will be omitted.")
        return gpd.GeoDataFrame(columns=["source_type", "source_id", "geometry"], geometry="geometry", crs=AREA_CRS)
    d = pd.read_csv(p, dtype={"source_id": "string"})
    needed = {"source_type", "source_id", "latitude", "longitude"}
    if not needed.issubset(d.columns):
        raise ValueError(f"Station CSV missing: {sorted(needed-set(d.columns))}")
    d["latitude"] = pd.to_numeric(d["latitude"], errors="coerce")
    d["longitude"] = pd.to_numeric(d["longitude"], errors="coerce")
    d = d[d.latitude.between(-90,90) & d.longitude.between(-180,180)].copy()
    d["source_type"] = d["source_type"].astype(str).str.lower().str.strip()
    d = d.drop_duplicates(["source_type", "source_id"])
    return gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d.longitude, d.latitude), crs="EPSG:4326").to_crs(AREA_CRS)


def river_lines(root, bounds, order_min):
    folder = root / "raw_data/HydroRIVERS_v10_af"
    gdb = folder / "HydroRIVERS_v10_af.gdb"
    if not gdb.exists():
        choices = list(folder.rglob("*.gdb")) if folder.exists() else []
        if not choices:
            warnings.warn("HydroRIVERS .gdb not found; blue river context omitted")
            return gpd.GeoDataFrame(geometry=[], crs=AREA_CRS)
        gdb = choices[0]
    # Only nearby reaches; prevents reading all ~1.5 million African segments.
    xmin,ymin,xmax,ymax = bounds
    from shapely.geometry import box
    region = gpd.GeoSeries([box(xmin-125000, ymin-125000, xmax+125000, ymax+125000)], crs=AREA_CRS)
    wgs_bbox = tuple(region.to_crs("EPSG:4326").total_bounds)
    g = gpd.read_file(gdb, layer="HydroRIVERS_v10_af", bbox=wgs_bbox)
    if g.crs is None:
        raise ValueError("HydroRIVERS layer has no CRS")
    field = next((x for x in ("ORD_STRA", "ORD_FLOW", "ORD_CLAS") if x in g), None)
    if field is None:
        warnings.warn("River order unavailable; skipping rivers instead of displaying every segment")
        return gpd.GeoDataFrame(geometry=[], crs=AREA_CRS)
    g = g[pd.to_numeric(g[field], errors="coerce").ge(order_min)]
    return g.to_crs(AREA_CRS)


def add_scalebar(ax, width_km=200):
    """Fixed-length scale bar inside geographic map, in equal-area CRS."""
    x0,x1=ax.get_xlim(); y0,y1=ax.get_ylim()
    length=width_km*1000
    if length > (x1-x0)*0.4: length=100000
    left=x0+(x1-x0)*0.66
    if left+length>x1: left=x1-length-(x1-x0)*.04
    bottom=y0+(y1-y0)*.045
    h=(y1-y0)*.009
    for i in range(4):
        seg=length/4
        ax.add_patch(Rectangle((left+i*seg,bottom),seg,h,fc="black" if i%2==0 else "white",ec=INK,lw=.8,zorder=20))
    for i in (0,2,4):
        ax.text(left+i*length/4,bottom+2.3*h,f"{int(i*length/4000)}",fontsize=8,ha="center",color=INK)
    ax.text(left+length+(x1-x0)*.008,bottom+2.3*h,"km",fontsize=8,color=INK)


def render_one(root, out, g, adm2, pts, rivers, metrics, kind, n_top=12):
    column = "mean_days_per_year_2000_2025" if kind == "historical" else "change_days_per_year"
    eligible = "eligible_hotspot" if kind == "historical" else "eligible_change"
    work = metrics[metrics[eligible] & metrics[column].notna()].sort_values([column,"adm3_id"], ascending=[False,True]).copy()
    if kind == "worsening":
        work=work[work[column]>0]
    ranked=work.head(n_top).copy()
    ranked.insert(0,"rank",range(1,len(ranked)+1))
    ranked[["rank","adm3_id","adm3_name","parent_adm2",column]].to_csv(
        out / ("01_top_historical_hotspots.csv" if kind=="historical" else "02_top_worsening_areas.csv"),index=False)
    geo=g.merge(metrics,on="adm3_id",how="left",validate="one_to_one")
    # A fixed schema is essential: never rely on automatic _x/_y suffixes.
    if "adm3_name" not in geo.columns:
        raise ValueError(f"ADM3 name missing after merge. Columns: {list(geo.columns)}")
    match=int(geo["adm3_name"].notna().sum())
    if match == 0 or match / len(geo) < .8:
        raise ValueError(f"ADM3 IDs failed to match annual metrics: {match}/{len(geo)}")
    country=adm2.dissolve()
    finite=geo.loc[geo[eligible].fillna(False),column].dropna().to_numpy(dtype=float)
    if not len(finite): raise ValueError("No eligible flood metric data")
    if kind=="historical":
        vmin,vmax=0,max(1,float(np.percentile(finite, 98)))
        cm=LinearSegmentedColormap.from_list("activity",[LIGHT,BLUE,LILAC,PURPLE])
        norm=Normalize(vmin,vmax)
        head="Historical Recorded Flood Activity"
        sub="2000–2025 | Mean annual recorded flood-activity days"
        cbar="Mean recorded activity (days/year)"
        table_value="Days/yr"
    else:
        bound=max(1,float(np.percentile(np.abs(finite),98)))
        cm=LinearSegmentedColormap.from_list("change",[(0,BLUE),(.5,LIGHT),(.77,LILAC),(1,PURPLE)])
        norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound)
        head="Worsening Recorded Flood Activity"
        sub="2019–2025 vs 2000–2018 | Change in mean annual recorded days"
        cbar="Change (days/year; recent − earlier)"
        table_value="Δ days/yr"
    fig=plt.figure(figsize=(19,10.5), facecolor=BG)
    # Independent panels prevent legends, colorbars and table text overlapping.
    ax=fig.add_axes([.032,.16,.635,.745])
    tb=fig.add_axes([.710,.22,.270,.66]); tb.set_axis_off()
    fig.text(.035,.957,f"South Sudan | {head} (ADM3)",fontsize=22,fontweight="bold",color=INK)
    fig.text(.035,.918,sub,fontsize=12.2,color="#61566E")
    geo.plot(ax=ax,color="#E5E5E5",edgecolor="white",linewidth=.17)
    eligible_geo=geo[geo[eligible].fillna(False) & geo[column].notna()]
    eligible_geo.plot(ax=ax,column=column,cmap=cm,norm=norm,edgecolor="white",linewidth=.12)
    adm2.boundary.plot(ax=ax,color="#625A64",linewidth=.55,zorder=5)
    country.boundary.plot(ax=ax,color=INK,linewidth=1.15,zorder=6)
    if not rivers.empty:
        rivers.plot(ax=ax,color=RIVER,linewidth=.58,alpha=.85,zorder=7)
    for typ,mark,color in [("dfo","^",DFO),("lake","o",LAKE)]:
        part=pts[pts.source_type.eq(typ)]
        if not part.empty:
            part.plot(ax=ax,marker=mark,color=color,markersize=44,edgecolor="white",linewidth=.5,zorder=11)
    markers=eligible_geo.merge(ranked[["adm3_id","rank"]],on="adm3_id",how="inner")
    for _,r in markers.iterrows():
        pt=r.geometry.representative_point()
        ax.annotate(str(int(r["rank"])),(pt.x,pt.y),ha="center",va="center",fontsize=8,color="white",weight="bold",
                    bbox=dict(boxstyle="circle,pad=.15",fc="#292534",ec="white",lw=.7),zorder=15)
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.annotate("N",xy=(.955,.94),xytext=(.955,.865),xycoords="axes fraction",textcoords="axes fraction",
                ha="center",fontsize=12,fontweight="bold",arrowprops=dict(arrowstyle="-|>",color=INK,lw=1.3))
    add_scalebar(ax)
    tb.set_title(f"Top {len(ranked)} {'historical hotspots' if kind=='historical' else 'increasing areas'}",
                 fontsize=14,fontweight="bold",loc="left",pad=14)
    show=ranked[["rank","adm3_name","parent_adm2",column]].copy()
    show[column]=show[column].map(lambda x:f"{x:+.1f}" if kind=="worsening" else f"{x:.1f}")
    show.columns=["#","ADM3","Parent ADM2",table_value]
    if not show.empty:
        table=tb.table(cellText=show.astype(str).values.tolist(),colLabels=list(show.columns),
                       cellLoc="left",colLoc="left",colWidths=[.08,.32,.36,.24],bbox=[0,.04,1,.88])
        table.auto_set_font_size(False);table.set_fontsize(9)
        for (i,j),cell in table.get_celld().items():
            cell.set_edgecolor("#DED9E7");cell.set_linewidth(.4)
            if i==0:cell.set_facecolor("#E8DCF4");cell.set_text_props(fontweight="bold")
            elif i%2==0:cell.set_facecolor("#F7F4FB")
    # Separate short color bar under right-side table.
    cax=fig.add_axes([.746,.153,.200,.016])
    bar=fig.colorbar(ScalarMappable(norm=norm,cmap=cm),cax=cax,orientation="horizontal")
    bar.ax.tick_params(labelsize=8)
    fig.text(.846,.126,cbar,ha="center",fontsize=9,color=INK)
    # Symbol-based legend UNDER map, not in map / on the colorbar.
    legend=[Line2D([0],[0],marker="^",color="none",markerfacecolor=DFO,markeredgecolor="white",markersize=9,label="DFO station"),
            Line2D([0],[0],marker="o",color="none",markerfacecolor=LAKE,markeredgecolor="white",markersize=9,label="Lake level"),
            Line2D([0],[0],color=RIVER,lw=2,label="HydroRIVERS"),
            Line2D([0],[0],color=INK,lw=1.5,label="ADM2 boundary")]
    fig.legend(handles=legend,loc="lower center",bbox_to_anchor=(.351,.080),ncol=4,frameon=False,fontsize=9)
    fig.text(.035,.038,
             "Flood activity records are not verified full-area inundation days. "
             "Hydrological points/river network provide geographic context only; source-to-region connections are UNVERIFIED.",
             fontsize=8.5,color="#55505D")
    path=out/("03_historical_activity_hydrology_map.png" if kind=="historical" else "04_worsening_activity_hydrology_map.png")
    pdf=path.with_suffix(".pdf")
    fig.savefig(path,dpi=250,facecolor=fig.get_facecolor())
    fig.savefig(pdf,facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"image":str(path),"matched_adm3":int(match),"numbered":int(len(ranked))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",default=str(root_default()))
    p.add_argument("--annual-panel",default=None)
    p.add_argument("--top-n",type=int,default=12)
    p.add_argument("--river-order-min",type=int,default=5)
    p.add_argument("--output",default=f"{BASE}/09c_flood_hydrology_overlay")
    args=p.parse_args()
    root=Path(args.root).resolve();out=root/args.output;out.mkdir(parents=True,exist_ok=True)
    annual=Path(args.annual_panel) if args.annual_panel else locate_annual(root)
    if not annual.is_absolute():annual=root/annual
    metrics=annual_metrics(annual)
    metrics.to_csv(out/"00_adm3_annual_activity_period_metrics.csv",index=False)
    a3=boundary(root,"ADM3");a2=boundary(root,"ADM2")
    pts=stations(root)
    rivers=river_lines(root,tuple(a2.total_bounds),args.river_order_min)
    print("HydroRIVERS context segments:",len(rivers))
    r1=render_one(root,out,a3,a2,pts,rivers,metrics,"historical",args.top_n)
    r2=render_one(root,out,a3,a2,pts,rivers,metrics,"worsening",args.top_n)
    method={"annual_source":str(annual),"periods":{"overall":"2000–2025","before":"2000–2018","after":"2019–2025"},
            "metric":"Mean annual days with recorded flood activity in ADM3",
            "change":"Mean annual days 2019–2025 minus 2000–2018",
            "observation_limitations":"Full daily observation denominator and permanent-water exclusion not independently verified",
            "waterway_limitations":"River and source positions are geographic context, not validated connections",
            "outputs":{"historical":r1,"worsening":r2}}
    (out/"05_methodology.json").write_text(json.dumps(method,indent=2,ensure_ascii=False))
    print("Saved maps and tables in:",out)

if __name__=="__main__":
    main()
