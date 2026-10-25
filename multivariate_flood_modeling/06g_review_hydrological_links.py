#!/usr/bin/env python3
"""06g: Review candidate hydrological connections and build approved monthly features.

Run without arguments from VS Code. This does NOT automatically approve links:
the first run makes a review CSV. Only rows explicitly approved WITH evidence
are used after subsequent runs.

Outputs: processed_data/multivariate_flood_modeling/06g_reviewed_hydrology/
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

KEY = ["source_type", "source_id", "level", "adm_id", "target_hybas_id"]
REVIEW = ["decision", "evidence_reference", "reviewer", "review_date", "connection_notes"]
APPROVED_VALUES = {"APPROVED"}
METRIC_CANDIDATES = {
    "lake": ["lake_level_mean", "mean_level", "water_level_mean", "value_mean",
             "monthly_mean", "mean", "level_mean"],
    "dfo": ["dfo_discharge_mean", "discharge_mean", "monthly_mean",
            "value_mean", "mean_discharge", "mean"],
}


def root_default():
    p = Path(__file__).resolve()
    return p.parents[2] if p.parent.name == "multivariate_flood_modeling" else Path.cwd()


def read_csv(path, required=None):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, dtype={"source_id": "string", "adm_id": "string"})
    missing = set(required or []) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    return df


def clean(df):
    df = df.copy()
    for c in ["source_type", "source_id", "level", "adm_id"]:
        df[c] = df[c].astype("string").str.strip()
    df["source_type"] = df["source_type"].str.lower()
    df["level"] = df["level"].str.upper()
    df["target_hybas_id"] = pd.to_numeric(df["target_hybas_id"], errors="coerce").astype("Int64").astype("string")
    return df


def input_paths(root, candidate_path=None):
    if candidate_path:
        return Path(candidate_path)
    base = root / "processed_data/multivariate_flood_modeling"
    for folder in ("06e_river_basin_connectivity", "06f_river_basin_connectivity"):
        p = base / folder / "04_downstream_admin_candidates_REVIEW.csv"
        if p.exists():
            return p
    raise FileNotFoundError("Cannot find 06e/06f downstream candidate CSV. Run 06e first.")


def create_review(base, out):
    p = out / "01_connection_review.csv"
    if not p.exists():
        df = clean(read_csv(base, KEY))
        for col in REVIEW:
            df[col] = ""
        df["decision"] = "PENDING"
        df = df.sort_values(["source_type", "source_id", "level", "adm_id", "downstream_hops"])
        df.to_csv(p, index=False)
        print("Created review template:", p)
    else:
        print("Preserved existing review edits:", p)
    return p


def validate(review, candidates):
    review = clean(review)
    candidates = clean(candidates)
    review["decision"] = review["decision"].fillna("").astype(str).str.strip().str.upper()
    for c in REVIEW[1:]:
        review[c] = review[c].fillna("").astype(str).str.strip()
    allowed = set(candidates[KEY].astype(str).itertuples(index=False, name=None))
    seen = set()
    reason = []
    for row in review.to_dict("records"):
        key = tuple(str(row[c]) for c in KEY)
        issues = []
        if key in seen:
            issues.append("DUPLICATE_KEY")
        seen.add(key)
        if key not in allowed:
            issues.append("NOT_IN_ORIGINAL_CANDIDATES")
        if row["decision"] not in {"APPROVED", "REJECTED", "PENDING"}:
            issues.append("BAD_DECISION")
        if row["decision"] == "APPROVED":
            if len(row["evidence_reference"]) < 12:
                issues.append("MISSING_EVIDENCE")
            if not row["reviewer"]:
                issues.append("MISSING_REVIEWER")
            if pd.isna(pd.to_datetime(row["review_date"], format="%Y-%m-%d", errors="coerce")):
                issues.append("BAD_DATE")
            if len(row["connection_notes"]) < 12:
                issues.append("MISSING_HYDROLOGICAL_REASON")
        reason.append("|".join(issues))
    review["validation_errors"] = reason
    review["verified_for_model"] = (review.decision == "APPROVED") & (review.validation_errors == "")
    return review


def identify_metric(df, typ):
    for c in METRIC_CANDIDATES[typ]:
        if c in df.columns:
            return c
    # Don't guess metric from a plausible but undocumented column.
    return None


def monthly_features(approved, monthly_path, output):
    """Build ADM-month feature table using approved source-to-admin mappings.

    No interpolation and no artificial zeros. If multiple approved sources cover
    one ADM region, source-specific columns are emitted (not averaged together).
    Monthly features are *contemporaneous observations*; models must LAG them
    to prediction time. No features are approved automatically.
    """
    if approved.empty:
        pd.DataFrame(columns=["level", "adm_id", "year", "month"]).to_csv(output, index=False)
        return {"status": "NO_APPROVED_LINKS", "rows": 0}
    if not monthly_path.exists():
        return {"status": "SOURCE_MONTHLY_NOT_FOUND", "rows": 0}
    m = read_csv(monthly_path, ["source_type", "source_id", "year", "month"])
    for c in ("source_type", "source_id"):
        m[c] = m[c].astype("string").str.strip()
    m["source_type"] = m["source_type"].str.lower()
    m["year"] = pd.to_numeric(m["year"], errors="coerce").astype("Int64")
    m["month"] = pd.to_numeric(m["month"], errors="coerce").astype("Int64")
    mapping = approved[["source_type", "source_id", "level", "adm_id"]].drop_duplicates()
    m = mapping.merge(m, on=["source_type", "source_id"], how="inner")
    if m.empty:
        return {"status": "NO_SOURCE_MATCH", "rows": 0}
    series = []
    not_found = []
    for typ in ("lake", "dfo"):
        part = m[m.source_type == typ].copy()
        if part.empty:
            continue
        metric = identify_metric(part, typ)
        if metric is None:
            not_found.append(f"{typ}: no recognized measurement field; columns={list(part.columns)}")
            continue
        part["measure"] = pd.to_numeric(part[metric], errors="coerce")
        part = part.dropna(subset=["year", "month", "measure"])
        part["feature"] = typ + "_" + part["source_id"].astype(str).str.replace(r"[^A-Za-z0-9]", "_", regex=True)
        grouped = part.groupby(["level", "adm_id", "year", "month", "feature"], dropna=False)["measure"].mean().reset_index()
        series.append(grouped)
    if not series:
        return {"status": "NO_RECOGNIZED_METRICS", "rows": 0, "details": not_found}
    long = pd.concat(series, ignore_index=True)
    wide = long.pivot_table(index=["level", "adm_id", "year", "month"],
                            columns="feature", values="measure", aggfunc="first").reset_index()
    wide.columns.name = None
    wide.to_csv(output, index=False)
    return {"status": "OK_SOURCE_FEATURES_UNLAGGED", "rows": int(len(wide)),
            "feature_columns": [c for c in wide.columns if c not in {"level","adm_id","year","month"}],
            "notes": "Do not use same-month observed values to predict that month. Use previous-month features in training."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(root_default()))
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--monthly-source", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    out = Path(args.output).resolve() if args.output else root / "processed_data/multivariate_flood_modeling/06g_reviewed_hydrology"
    out.mkdir(parents=True, exist_ok=True)
    source = input_paths(root, args.candidates)
    candidates = read_csv(source, KEY)
    review_file = create_review(source, out)
    reviewed = read_csv(review_file, KEY + REVIEW)
    evaluated = validate(reviewed, candidates)
    approved = evaluated[evaluated.verified_for_model].copy()
    evaluated.to_csv(out / "02_all_reviews_validated.csv", index=False)
    approved.to_csv(out / "03_approved_connections.csv", index=False)
    evaluated[~evaluated.verified_for_model].to_csv(out / "04_not_approved.csv", index=False)
    monthly_path = (Path(args.monthly_source).resolve() if args.monthly_source else
                    root / "processed_data/multivariate_flood_modeling/06a_lake_dfo_monthly/station_lake_monthly.csv")
    feature_info = monthly_features(approved, monthly_path, out / "05_approved_monthly_features.csv")
    summary = {"candidate_rows": int(len(candidates)), "review_rows": int(len(reviewed)),
               "approved": int(len(approved)), "pending": int((evaluated.decision == "PENDING").sum()),
               "invalid_or_rejected": int(((evaluated.decision != "PENDING") & ~evaluated.verified_for_model).sum()),
               "monthly_feature_build": feature_info,
               "review_file": str(review_file),
               "warning": "Only human-documented approvals included; connected hydrology is not automatically a validated causal or useful predictor."}
    (out / "06_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf8")
    (out / "README.txt").write_text(
        "06g workflow\n"
        "1. Run this script once. It creates 01_connection_review.csv.\n"
        "2. Review river-snapping, river direction, basin topology, source location and evidence independently.\n"
        "3. Set decision to APPROVED for verified connections only; fill evidence_reference, reviewer, review_date (YYYY-MM-DD), connection_notes.\n"
        "4. Re-run: edited review rows are PRESERVED. Unverified links stay out of feature output.\n"
        "5. 05_approved_monthly_features.csv contains OBSERVED monthly values, not safe forecast inputs until lagged.\n"
        "6. You still need an explicit integration step to merge these columns with 05a monthly panel and run 06b.\n"
        "7. HydroBASINS candidate down-chain does not by itself prove station-to-region flood propagation.\n",
        encoding="utf8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
