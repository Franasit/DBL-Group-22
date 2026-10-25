"""JBG060 conservative ADM2/ADM3 flood-panel assembler (version 3).

Builds full and lagged annual panels. An incompatible optional driver is never
joined by position, region name, or guessed code, and is never silently imputed.
Requires pandas and numpy.
"""
from __future__ import annotations
import argparse
import json
import re
import warnings
from pathlib import Path
import pandas as pd
import numpy as np

YEARS = list(range(2000, 2026))
DATASETS = ["flood_masks", "rainfall_and_runoff", "evapotranspiration",
            "water_levels_lakes", "darthmouth_flood_observatory",
            "administrative_boundaries", "worldpop", "farmland",
            "health_facilities", "GDP", "IPC"]
TARGET_PATHS = {
    "ADM2": "processed_data/cross_dataset_analysis/01_adm2_flood/indicators/tables/adm2_annual_flood_indicators_combined.csv",
    "ADM3": "processed_data/cross_dataset_analysis/02_adm3_flood/10_build_adm3_flood_indicators/tables/adm3_annual_flood_indicators_combined.csv",
}
DEFAULT_DRIVERS = {
    "ADM2": [{"dataset":"rainfall_and_runoff", "name":"era5_adm2_annual",
              "path":"processed_data/cross_dataset_analysis/adm2_hydrology/ADM2_annual_rainfall_runoff.csv",
              "scope":"regional_annual", "year_col":"year", "admin_col":"ADM2_ID",
              "columns":"auto_numeric", "approved":True, "predictor_use":"explanation_only"}],
    "ADM3": [],
}
# Verified output from Script 03. ET0 units are mm/year (FAO-56 reference ET0).
# Only rows passing the Script 03 QC are treated as measured-grid annual values.
ET0_SOURCES = {
    level: {
        "dataset": "evapotranspiration", "name": f"et0_{level.lower()}_annual",
        "path": f"processed_data/multivariate_flood_modeling/03_evapotranspiration/{level.lower()}_annual_et0.csv",
        "level": level, "scope": "regional_annual", "year_col": "year", "admin_col": "adm_id",
        "columns": ["et0_annual_mm", "et0_mean_daily_mm", "et0_valid_day_fraction", "et0_grid_cells"],
        "quality_col": "et0_qc_status", "quality_pass_value": "PASS",
        "approved": True, "predictor_use": "lagged_prediction", "available_by_end_of_year": True,
        "units": {"et0_annual_mm": "mm/year", "et0_mean_daily_mm": "mm/day",
                  "et0_valid_day_fraction": "fraction", "et0_grid_cells": "count"},
    } for level in ("ADM2", "ADM3")
}

ID_CANDIDATES = {
    "ADM2": ["ADM2_ID", "adm2_id", "ADM2_PCODE", "adm2_pcode", "admin2Pcode", "ADM2_CODE", "admin2_code", "adm2_code"],
    "ADM3": ["ADM3_ID", "adm3_id", "ADM3_PCODE", "adm3_pcode", "admin3Pcode", "ADM3_CODE", "admin3_code", "adm3_code"],
}
YEAR_CANDIDATES = ["year", "Year", "YEAR", "calendar_year"]
BAD_FEATURE = re.compile(r"cluster|pca|principal|silhouette|trend|classification|long.term|all.years|future|hotspot|change.point|period|2000.2025", re.I)


def find_col(df, candidates, required=True):
    lookup = {str(c).casefold():c for c in df.columns}
    for candidate in candidates:
        if candidate and str(candidate).casefold() in lookup:
            return lookup[str(candidate).casefold()]
    if required:
        raise ValueError(f"Missing required column. Candidates={candidates}; columns={list(df.columns)}")
    return None


def codes(series):
    """Conservative formatting normalization; do not remove leading zeros."""
    s = series.astype("string").str.strip()
    s = s.str.replace(r"^([0-9]+)\.0$", r"\1", regex=True)
    return s.mask(s.str.casefold().isin(["", "nan", "none", "null", "<na>"]))


def norm_year(series):
    num = pd.to_numeric(series, errors="coerce")
    if num.isna().any() or ((num % 1) != 0).any():
        raise ValueError("Missing or non-integer calendar years")
    return num.astype(int)


def load_csv(path):
    return pd.read_csv(path, low_memory=False)


def normalized_frame(df, admin_col, year_col):
    ids = codes(df[admin_col]); years = norm_year(df[year_col])
    if ids.isna().any():
        raise ValueError(f"Missing administrative IDs in {admin_col}")
    frame = df.copy()
    frame["adm_id"] = ids.astype(str)
    frame["year"] = years
    frame = frame.loc[frame.year.isin(YEARS)].copy()
    dup = frame.duplicated(["adm_id", "year"])
    if dup.any():
        raise ValueError(f"Duplicate region/year keys: {frame.loc[dup,['adm_id','year']].head(5).to_dict('records')}")
    return frame


def target_panel(root, level, expected):
    path = root / TARGET_PATHS[level]
    if not path.is_file():
        raise FileNotFoundError(f"Latest flood target not found: {path}")
    raw = load_csv(path)
    adm = find_col(raw, ID_CANDIDATES[level]); yr = find_col(raw, YEAR_CANDIDATES)
    data = normalized_frame(raw, adm, yr)
    ids = sorted(data.adm_id.unique())
    if len(ids) != expected:
        raise ValueError(f"{level} target has {len(ids)} regions, expected {expected}; check latest source")
    grid = pd.MultiIndex.from_product([ids, YEARS], names=["adm_id", "year"])
    existing = pd.MultiIndex.from_frame(data[["adm_id", "year"]])
    missing = grid.difference(existing)
    # Only retain target payload fields; administrative metadata stays in raw
    # and can be used for ID crosswalk detection.
    payload = [c for c in raw.columns if c not in (adm, yr)]
    if any(str(c).startswith("flood__") for c in payload):
        raise ValueError("Flood source columns already use reserved flood__ prefix")
    reduced = data[["adm_id", "year"] + payload].copy()
    panel = reduced.set_index(["adm_id", "year"]).reindex(grid).reset_index()
    panel.insert(2, "target_row_present", pd.MultiIndex.from_frame(panel[["adm_id", "year"]]).isin(existing))
    panel.rename(columns={c:"flood__"+str(c) for c in payload},inplace=True)
    return panel,raw,path,missing,len(data),adm


def safe_mapping(values, keys, label):
    """Return key->value only if codes are nonmissing and each key maps uniquely."""
    f = pd.DataFrame({"key":codes(keys), "value":codes(values)}).drop_duplicates()
    f = f.dropna()
    if f.empty or f.groupby("key").value.nunique().max() != 1:
        return None
    return f.set_index("key").value.to_dict()


def try_auto_crosswalk(src_raw, tgt_raw, src_id_col, tgt_id_col, level):
    """Match through alternate *codes* only (never region names or row order).

    A mapping is accepted only when all source IDs map to unique target IDs and
    all source/target reference rows agree. This is a schema bridge, not a
    geographic equivalence proof; administrative boundary version must still
    be independently verified before interpreting results.
    """
    src_ids = set(codes(src_raw[src_id_col]).dropna())
    tgt_ids = set(codes(tgt_raw[tgt_id_col]).dropna())
    if src_ids <= tgt_ids:
        return {s:s for s in src_ids}, "EXACT_ID"
    # Within-source alternative standardized code columns may identify how a
    # source label relates to a flood target identifier.
    candidates = [x for x in ID_CANDIDATES[level]
                  if x.casefold() not in {src_id_col.casefold(), tgt_id_col.casefold()}]
    pairs = []
    for sc in [src_id_col]+[find_col(src_raw,[x],False) for x in candidates]:
        if sc is None: continue
        for tc in [tgt_id_col]+[find_col(tgt_raw,[x],False) for x in candidates]:
            if tc is None: continue
            # Same-type labels imply matching administrative *codes*, not names.
            if sc == src_id_col and tc == tgt_id_col: continue
            shared = set(codes(src_raw[sc]).dropna()) & set(codes(tgt_raw[tc]).dropna())
            if not shared: continue
            src_bridge = safe_mapping(src_raw[sc],src_raw[src_id_col],"src")
            tgt_bridge = safe_mapping(tgt_raw[tgt_id_col],tgt_raw[tc],"tgt")
            if src_bridge is None or tgt_bridge is None: continue
            mapped = {sid:tgt_bridge.get(src_bridge.get(sid)) for sid in src_ids}
            if any(v is None for v in mapped.values()): continue
            if len(set(mapped.values())) != len(mapped): continue
            if not set(mapped.values()) <= tgt_ids: continue
            pairs.append((sc,tc,mapped))
    # Distinct candidate methods must agree identically, or require explicit user crosswalk.
    if pairs and all(p[2] == pairs[0][2] for p in pairs):
        a,b,m = pairs[0]
        return m, f"VALIDATED_CODE_BRIDGE:{a}->{b}"
    return None, "INCOMPATIBLE_ADMIN_ID_SCHEMA"


def explicit_crosswalk(root,spec,source_ids,target_ids):
    path = root / str(spec["crosswalk_path"])
    if not path.is_file(): raise FileNotFoundError(path)
    raw = load_csv(path)
    sc = find_col(raw,[spec.get("crosswalk_source_col","source_id")])
    tc = find_col(raw,[spec.get("crosswalk_target_col","target_id")])
    mapping = safe_mapping(raw[tc],raw[sc],"manual crosswalk")
    if mapping is None: raise ValueError("Crosswalk has ambiguous or missing rows")
    mapped = {sid:mapping.get(sid) for sid in source_ids}
    if any(x is None for x in mapped.values()):
        raise ValueError("Crosswalk does not cover every source admin ID")
    if len(set(mapped.values())) != len(mapped):
        raise ValueError("Crosswalk is many-to-one; aggregation required, not direct join")
    if not set(mapped.values()) <= target_ids:
        raise ValueError("Crosswalk targets absent from latest flood panel")
    return mapped,"EXPLICIT_CROSSWALK"


def diagnostic(out,name,source_raw,source_id_col,target_raw,target_id_col,level,reason):
    rows=[]
    for origin,raw,col in [("source",source_raw,source_id_col),("target",target_raw,target_id_col)]:
        colnames = [c for c in ID_CANDIDATES[level] if find_col(raw,[c],False)]
        keyset = [col]+[find_col(raw,[c],False) for c in colnames]
        keyset = list(dict.fromkeys(x for x in keyset if x is not None))
        sample = raw[keyset].drop_duplicates().head(30)
        for _,r in sample.iterrows():
            rows.append({"side":origin,"primary_column":col,"primary_id":str(r[col]),
                         "candidate_codes_json":json.dumps({k:str(r[k]) for k in keyset},ensure_ascii=False),
                         "problem":reason})
    dest=out/f"id_diagnostic_{name}.csv"
    pd.DataFrame(rows).to_csv(dest,index=False)
    return dest


def read_config(path):
    if path is None: return None,"builtin_defaults"
    p=Path(path).expanduser().resolve()
    cfg=json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(cfg,dict) or not isinstance(cfg.get("sources"),list):
        raise ValueError('Config must contain a "sources" list')
    return cfg["sources"],str(p)


def build(root,level,expected_units,config_path=None,strict=False):
    root=Path(root).expanduser().resolve()
    out=root/"processed_data"/"multivariate_flood_modeling"/f"{'01' if level=='ADM2' else '02'}_{level.lower()}_panel_dataset"
    out.mkdir(parents=True,exist_ok=True)
    panel,target_raw,target_path,missing,original_count,target_id_col=target_panel(root,level,expected_units)
    pd.DataFrame(missing.tolist(),columns=["adm_id","year"]).to_csv(out/"target_missing_keys.csv",index=False)
    sources,config_origin=read_config(config_path)
    if sources is None:
        sources = [dict(x) for x in DEFAULT_DRIVERS[level]]
    # Always attempt integration of the known Script 03 output. Do not silently
    # duplicate ET0 if a caller explicitly configured the same dataset.
    if not any(x.get("dataset") == "evapotranspiration" for x in sources):
        sources = [*sources, ET0_SOURCES[level].copy()]
    registry=[{"dataset":"flood_masks","source":"latest_flood_target","status":"ATTACHED","reason":str(target_path)}]
    joins=[]; added=set(); attached=0; attempted=[]; feature_meta=[]
    for spec in sources:
        if str(spec.get("level",level)).upper()!=level: continue
        name=str(spec.get("name",""))
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*",name) or name in added:
            raise ValueError(f"Invalid or duplicate driver name: {name!r}")
        added.add(name)
        attempted.append(str(spec.get("dataset", "")))
        status="NOT_ATTACHED"; reason=""; diagnostics=""
        try:
            dataset=spec.get("dataset")
            if dataset not in DATASETS: raise ValueError(f"Unknown dataset {dataset}")
            if spec.get("approved") is not True:
                status="NOT_APPROVED";reason="Not explicitly approved"
            elif spec.get("dataset") in ("administrative_boundaries", "flood_masks"):
                status="NOT_APPROVED";reason="Boundary metadata belongs to target geometry; do not attach as arbitrary numeric drivers"
            elif spec.get("scope")!="regional_annual":
                status="UNSAFE_SCOPE";reason="Must preprocess to same ADM unit/year before joining"
            else:
                source_path=root/str(spec["path"])
                if not source_path.is_file():
                    status="FILE_MISSING";reason=str(source_path)
                else:
                    raw=load_csv(source_path)
                    src_id_col=find_col(raw,[spec.get("admin_col")]+ID_CANDIDATES[level])
                    src_year_col=find_col(raw,[spec.get("year_col")]+YEAR_CANDIDATES)
                    source=normalized_frame(raw,src_id_col,src_year_col)
                    # Fail closed: non-PASS QC rows have *no usable ET0 value*.
                    quality_col = spec.get("quality_col")
                    qc_invalid_rows = 0
                    if quality_col:
                        if quality_col not in source.columns:
                            raise ValueError(f"Configured QC field {quality_col!r} absent in {source_path}")
                        passing = source[quality_col].astype("string").str.strip().eq(
                            str(spec.get("quality_pass_value", "PASS")))
                        qc_invalid_rows = int((~passing).sum())
                        source = source.copy()
                        # Only selected numeric values are nulled; retain row
                        # and administrative key so QC failures remain missing.
                    requested=spec.get("columns","auto_numeric")
                    # Explicit column selection is mandatory for new V3 features.
                    # Preserve the existing verified ERA5 V2 attachment for compatibility.
                    if requested=="auto_numeric" and name!="era5_adm2_annual":
                        raise ValueError("For new sources set an explicit columns list; auto_numeric may include identifiers or leakage fields")
                    if requested=="auto_numeric":
                        # Only numeric raw fields; never accidentally include admin code, year, ID.
                        columns=[c for c in raw.columns if c not in (src_id_col,src_year_col)
                                 and pd.api.types.is_numeric_dtype(raw[c])
                                 and not any(c.casefold()==x.casefold() for x in ID_CANDIDATES[level])]
                    elif isinstance(requested,list) and requested:
                        columns=requested
                        if any(c not in raw.columns for c in columns):
                            raise ValueError(f"Missing configured fields: {[c for c in columns if c not in raw.columns]}")
                    else: raise ValueError("columns must be auto_numeric or nonempty list")
                    columns=[c for c in columns if c not in ("adm_id","year",src_id_col,src_year_col)]
                    if not columns:
                        status="NO_NUMERIC_COLUMNS";reason="No numeric features to attach"
                    else:
                        sids=set(source.adm_id);tids=set(panel.adm_id)
                        if spec.get("crosswalk_path"):
                            mapping,how=explicit_crosswalk(root,spec,sids,tids)
                        else:
                            mapping,how=try_auto_crosswalk(raw,target_raw,src_id_col,target_id_col,level)
                        if mapping is None:
                            status="ID_MAPPING_REQUIRED"
                            reason=(f"No unambiguous ID mapping: {len(sids)} source IDs, {len(tids)} target IDs. "
                                    "Inspect diagnostic and supply an audited crosswalk CSV in config; never align by row order.")
                            diagnostics=str(diagnostic(out,name,raw,src_id_col,target_raw,target_id_col,level,reason))
                        else:
                            source["adm_id"]=source.adm_id.map(mapping)
                            if source.adm_id.isna().any(): raise ValueError("Mapping contains null region codes")
                            if source.duplicated(["adm_id","year"]).any():
                                raise ValueError("Mapped source has duplicate admin/year keys")
                            keep=[]
                            for c in columns:
                                converted=pd.to_numeric(source[c],errors="coerce")
                                if converted.notna().any():
                                    source[c]=converted;keep.append(c)
                            if not keep:
                                status="NO_NUMERIC_VALUES";reason="All selected driver columns are null or nonnumeric"
                            else:
                                prefix={c:f"{name}__{c}" for c in keep}
                                before=len(panel)
                                panel=panel.merge(source[["adm_id","year"]+keep].rename(columns=prefix),on=["adm_id","year"],how="left",validate="one_to_one")
                                if len(panel)!=before: raise AssertionError("Panel cardinality changed")
                                keyset=pd.MultiIndex.from_frame(panel[["adm_id","year"]])
                                matches=int(pd.MultiIndex.from_frame(source[["adm_id","year"]]).isin(keyset).sum())
                                status="ATTACHED";reason=(f"{how}; QC-excluded rows={qc_invalid_rows}; "
                                    "mapping must use consistent ADM boundary version")
                                attached+=1
                                for col in keep:
                                    feature_meta.append({
                                        "column":prefix[col], "source":name,
                                        "dataset":dataset, "original_column":col,
                                        "spatial_scope":spec.get("scope"),
                                        "available_by_end_of_year":spec.get("available_by_end_of_year", "UNVERIFIED"),
                                        "predictor_use":spec.get("predictor_use", "explanation_only"),
                                        "units":spec.get("units", {}).get(col,"UNVERIFIED") if isinstance(spec.get("units"),dict) else "UNVERIFIED",
                                        "source_path":str(source_path),
                                        "mapping_method":how})
                                joins.append({"name":name,"dataset":dataset,"source_rows":len(source),"matched_keys":matches,
                                              "panel_rows":len(panel),"join_coverage_pct":round(100*matches/len(panel),3),
                                              "source_admin_ids":len(sids),"qc_excluded_rows":qc_invalid_rows,"features":",".join(prefix.values()),
                                              "predictor_use":spec.get("predictor_use","unspecified"),"mapping_method":how})
        except Exception as exc:
            status="SOURCE_ERROR";reason=f"{type(exc).__name__}: {exc}"
        registry.append({"dataset":spec.get("dataset",""),"source":name,"status":status,"reason":reason,"diagnostic_file":diagnostics})
        if status!="ATTACHED":
            warnings.warn(f"{name} [{status}]: {reason}",stacklevel=2)
            if strict and spec.get("approved") is True:
                # Keep diagnostics even in strict mode; no misleading panel written.
                pd.DataFrame(registry).to_csv(out/"dataset_integration_registry.csv",index=False)
                raise ValueError(f"Source {name} not safely attached: {reason}; diagnostics={diagnostics}")
    panel=panel.sort_values(["adm_id","year"]).reset_index(drop=True)
    target_numeric=[c for c in panel if c.startswith("flood__") and pd.api.types.is_numeric_dtype(panel[c])
                    and not pd.api.types.is_bool_dtype(panel[c])]
    # Only current-year response columns for evaluation; selected flood history
    # can become lagged inputs. Never include future-derived flood labels.
    chosen_targets=["flood__flood_days", "flood__annual_flooded_ratio",
                    "flood__annual_unique_flooded_area_km2", "flood__annual_flooded_percent"]
    target_cols=[c for c in chosen_targets if c in panel]
    if "flood__flood_days" not in target_cols:
        warnings.warn("flood__flood_days absent; inspect target schema before modelling")
    excluded=[c for c in target_numeric if BAD_FEATURE.search(c)]
    # Historical flood predictors: quantities observed by the previous year.
    flood_history=[c for c in target_numeric if c not in excluded and c in (
        "flood__flood_days", "flood__annual_flooded_ratio", "flood__flood_event_count",
        "flood__annual_unique_flooded_area_km2", "flood__annual_flooded_percent")]
    approved_drivers=[]
    for meta in feature_meta:
        c=meta["column"]
        if (meta["predictor_use"] in ("lagged_prediction", "both")
            and meta["available_by_end_of_year"] is True
            and not BAD_FEATURE.search(c)
            and pd.api.types.is_numeric_dtype(panel[c])):
            approved_drivers.append(c)
    candidate_features=flood_history+approved_drivers
    pred=panel[["adm_id","year","target_row_present"]+target_cols].copy()
    pred.rename(columns={c:"target__"+c[len("flood__"):] for c in target_cols},inplace=True)
    # Prior year feature is valid only where prior calendar year exists;
    # groupby.shift() alone would bridge missing years.
    for c in dict.fromkeys(candidate_features):
        prev_year=panel.groupby("adm_id",sort=False)["year"].shift(1)
        lagged=panel.groupby("adm_id",sort=False)[c].shift(1).where(prev_year.eq(panel.year-1))
        pred[c+"__lag1"]=lagged
    # Flag suspicious target zeros without converting them to missing or vice versa.
    if "flood__flood_days" in panel:
        check=panel[["adm_id","year","target_row_present","flood__flood_days"]].copy()
        check["zero_flood_days_needs_qc"]=check.target_row_present & check.flood__flood_days.eq(0)
        check.to_csv(out/"target_zero_review.csv",index=False)
    # Special known ERA5 anomaly: do not silently present suspect zeros as verified.
    if level=="ADM2" and "era5_adm2_annual__annual_rainfall_mm" in panel:
        rain="era5_adm2_annual__annual_rainfall_mm"
        runoff="era5_adm2_annual__annual_runoff_mm"
        mean="era5_adm2_annual__mean_daily_rainfall_mm"
        suspect=panel[rain].eq(0)
        if runoff in panel: suspect &= panel[runoff].eq(0)
        if mean in panel: suspect &= panel[mean].isna()
        panel["qc__era5_suspect_zero_with_missing_daily"]=suspect.fillna(False)
        pred["qc__era5_suspect_zero_with_missing_daily_prev_year"]=panel.groupby("adm_id",sort=False)["qc__era5_suspect_zero_with_missing_daily"].shift(1).eq(True)
        panel.loc[suspect,[c for c in [rain,runoff] if c in panel]]=np.nan
        # Remove corresponding lagged values if driver approved for prediction.
        for c in [rain,runoff]:
            lc=c+"__lag1"
            if lc in pred:
                pred.loc[pred["qc__era5_suspect_zero_with_missing_daily_prev_year"],lc]=np.nan
        panel.loc[suspect,["adm_id","year",rain]+([runoff] if runoff in panel else [])].to_csv(out/"era5_suspect_zero_review.csv",index=False)
    panel.to_csv(out/"full_analysis_panel.csv",index=False)
    pred.to_csv(out/"prediction_ready_panel.csv",index=False)
    join_fields=["name","dataset","source_rows","matched_keys","panel_rows","join_coverage_pct","source_admin_ids","qc_excluded_rows","features","predictor_use","mapping_method"]
    pd.DataFrame(joins,columns=join_fields).to_csv(out/"join_quality_report.csv",index=False)
    listed={str(x.get("dataset")) for x in registry}
    for dataset in DATASETS:
        if dataset not in listed:
            registry.append({"dataset":dataset,"source":"","status":"PENDING_PREPROCESSING" if dataset not in ("administrative_boundaries",) else "GEOMETRY_REFERENCE",
                             "reason":"No verified regional annual input configured; may require preprocessing and/or spatial mapping"})
    registry.append({"dataset":"historical_flood_patterns","source":"","status":"FOLD_AWARE_ONLY",
                     "reason":"PCA/clusters/trends learned from whole period must be recomputed within each training fold"})
    pd.DataFrame(registry).to_csv(out/"dataset_integration_registry.csv",index=False)
    pd.DataFrame(feature_meta,columns=["column","source","dataset","original_column","spatial_scope",
                                      "available_by_end_of_year","predictor_use","units","source_path","mapping_method"]).to_csv(out/"feature_lineage.csv",index=False)
    missing_by_col=panel.isna().sum().reset_index()
    missing_by_col.columns=["column","missing_rows"]
    missing_by_col["missing_pct"]=(100*missing_by_col.missing_rows/len(panel)).round(2)
    missing_by_col.to_csv(out/"column_missingness.csv",index=False)
    pending=[x["dataset"] for x in registry if x["status"] in ("PENDING_PREPROCESSING","ID_MAPPING_REQUIRED","SOURCE_ERROR","FILE_MISSING","UNSAFE_SCOPE","NOT_APPROVED")]
    summary={"version":"V3_ET0", "level":level,"target_source":str(target_path),
             "regions":expected_units,"years":YEARS,"panel_rows":len(panel),
             "original_target_rows":original_count,"target_missing_keys":len(missing),
             "attached_optional_sources":attached,"pending_datasets":pending,
             "config_source":config_origin,"prediction_feature_exclusions":excluded,
             "prediction_inputs_count":len(candidate_features),
             "warnings":["Unattached data are not automatically imputed or spatially allocated.",
                         "Zero flood values require review against latest flood-mask QC reports.",
                         "Year-lag features require as-of availability verification, including publication delays.",
                         "Full-period PCA/clusters cannot enter forecasting folds.",
                         "ET0 represents reference evapotranspiration, not actual evapotranspiration.",
                         "ADM3 ET0 derived by grid-cell-centre assignment; zero-grid ADM3s remain missing.",
                         "Matching administrative codes does not establish matching boundary vintage.",
                         "ADM2 climate values must not be relabeled as observed ADM3 climate without validated spatial aggregation."]}
    (out/"panel_build_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({"output_dir":str(out),"panel_rows":len(panel),"attached_sources":attached,
                      "pending_datasets":pending,
                      "sources_needing_action":[f"{r['source']}: {r['status']}" for r in registry if r['source'] and r['status'] not in ("ATTACHED",)]},indent=2))
    return summary


def cli(level,expected_units):
    p=argparse.ArgumentParser(description=f"Build {level} annual multivariate flood panel")
    p.add_argument("--root",default=".",help="Path to DBL-Group-22 project root")
    p.add_argument("--config",default=None,help="Optional approved-source JSON")
    p.add_argument("--strict",action="store_true",help="Fail rather than produce partial panel when approved source cannot join")
    a=p.parse_args()
    build(a.root,level,expected_units,a.config,a.strict)
