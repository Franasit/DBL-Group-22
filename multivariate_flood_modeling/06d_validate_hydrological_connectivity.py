#!/usr/bin/env python3
"""
06d_validate_hydrological_connectivity.py

Create an auditable review workflow from 06c's geographic candidates.
Important: geographic proximity is NOT hydrological connectivity.
No candidate is approved automatically.

Inputs:
    processed_data/multivariate_flood_modeling/06c_hydrological_spatial_mapping/
        01_hydrology_source_locations.csv
        02_adm2_geographic_candidates_REVIEW.csv
        03_adm3_geographic_candidates_REVIEW.csv
    processed_data/multivariate_flood_modeling/06a_lake_dfo_monthly/
        station_lake_monthly.csv  (optional, for time coverage)

Outputs under 06d_hydrological_connectivity:
    01_candidate_review_template.csv     (written once, preserved on reruns)
    02_reviewed_mapping_APPROVED.csv      (only human-verified rows)
    03_pending_review.csv
    04_rejected_or_invalid.csv
    05_source_coverage.csv
    06_validation_summary.json
    07_review_instructions.md

Workflow:
  1. Run: python scripts/multivariate_flood_modeling/06d_validate_hydrological_connectivity.py
  2. Open 01_candidate_review_template.csv; fill review_status,
     relationship, evidence_source, reviewer_name, review_date,
     and optionally hydrological_notes for each supported relationship.
     Review must be based on external waterway/catchment evidence.
  3. Run again. Only APPROVED rows passing strict validation are exported.
  4. 02_reviewed_mapping_APPROVED.csv has the same five columns as 06a's
     hydrology_adm_mapping_TEMPLATE.csv, with approved relations only.

The script does not infer causality, discharge travel time, or forecast
availability from station positions. Lag choice belongs to the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CANDIDATE_COLUMNS = [
    "source_type", "source_id", "level", "adm_id",
    "rank_by_distance", "distance_to_polygon_km_approx",
    "station_inside_polygon", "candidate_method",
    "review_status", "relationship", "evidence_source",
    "reviewer_name", "review_date", "hydrological_notes",
]
KEY_COLUMNS = ["source_type", "source_id", "level", "adm_id"]
MAPPING_COLUMNS = ["source_type", "source_id", "level", "adm_id", "relationship"]
VALID_RELATIONSHIPS = {
    "UPSTREAM_INFLOW", "DOWNSTREAM_REACH", "SAME_RIVER_REACH",
    "CONNECTED_LAKE_OUTFLOW", "SHARED_CATCHMENT", "OTHER_DOCUMENTED",
}


def project_root() -> Path:
    # In DBL-Group-22/scripts/multivariate_flood_modeling/<file>.py
    here = Path(__file__).resolve()
    if here.parent.name == "multivariate_flood_modeling" and here.parent.parent.name == "scripts":
        return here.parent.parent.parent
    return Path.cwd().resolve()


def require_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}\nRun Script 06c first.")
    return pd.read_csv(path, dtype={"source_id": "string", "adm_id": "string"})


def clean_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in KEY_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        df[col] = df[col].astype("string").str.strip()
    df["source_type"] = df["source_type"].str.lower()
    df["level"] = df["level"].str.upper()
    return df


def assemble_candidates(input_dir: Path) -> pd.DataFrame:
    result = []
    for level, name in [
        ("ADM2", "02_adm2_geographic_candidates_REVIEW.csv"),
        ("ADM3", "03_adm3_geographic_candidates_REVIEW.csv"),
    ]:
        df = clean_key(require_csv(input_dir / name))
        if not df.empty and not df["level"].eq(level).all():
            raise ValueError(f"Unexpected ADM level in {name}")
        df = df.drop(columns=["hydrological_connection_verified"], errors="ignore")
        for col in CANDIDATE_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        df["review_status"] = "PENDING"
        for col in ("relationship", "evidence_source", "reviewer_name", "review_date", "hydrological_notes"):
            df[col] = ""
        result.append(df[CANDIDATE_COLUMNS])
    combined = pd.concat(result, ignore_index=True)
    if combined.duplicated(KEY_COLUMNS).any():
        print("WARNING: duplicate candidate keys; keeping one per unique key.")
        combined = combined.drop_duplicates(KEY_COLUMNS)
    return combined


def make_review_file(template: Path, candidates: pd.DataFrame) -> None:
    if template.exists():
        print(f"Keeping existing review file (not overwriting): {template}")
        return
    candidates.to_csv(template, index=False)
    print(f"Created review template: {template}")


def evaluate_review(
    reviewed: pd.DataFrame,
    current_candidates: pd.DataFrame,
) -> pd.DataFrame:
    reviewed = clean_key(reviewed)
    for col in CANDIDATE_COLUMNS:
        if col not in reviewed.columns:
            reviewed[col] = ""
    reviewed = reviewed[CANDIDATE_COLUMNS].copy()
    reviewed = reviewed.fillna("")
    for col in ("review_status", "relationship", "evidence_source", "reviewer_name",
                "review_date", "hydrological_notes"):
        reviewed[col] = reviewed[col].astype(str).str.strip()
    reviewed["review_status"] = reviewed["review_status"].str.upper()
    reviewed["relationship"] = reviewed["relationship"].str.upper()

    allowed = set(tuple(x) for x in current_candidates[KEY_COLUMNS].itertuples(index=False, name=None))
    seen = set()
    reasons = []
    for r in reviewed.itertuples(index=False):
        d = r._asdict()
        k = tuple(d[c] for c in KEY_COLUMNS)
        errors = []
        if k in seen:
            errors.append("DUPLICATE_REVIEW_KEY")
        seen.add(k)
        if k not in allowed:
            errors.append("KEY_NOT_IN_06C_CANDIDATES")
        status = d["review_status"]
        if status not in {"PENDING", "APPROVED", "REJECTED"}:
            errors.append("INVALID_STATUS")
        if status == "APPROVED":
            if d["relationship"] not in VALID_RELATIONSHIPS:
                errors.append("MISSING_OR_INVALID_RELATIONSHIP")
            if len(d["evidence_source"]) < 10:
                errors.append("MISSING_EVIDENCE_SOURCE")
            if not d["reviewer_name"]:
                errors.append("MISSING_REVIEWER")
            try:
                if pd.isna(pd.to_datetime(d["review_date"], format="%Y-%m-%d", errors="coerce")):
                    errors.append("INVALID_REVIEW_DATE")
            except Exception:
                errors.append("INVALID_REVIEW_DATE")
            if not d["hydrological_notes"]:
                errors.append("MISSING_HYDROLOGICAL_NOTES")
        reasons.append("|".join(errors))
    reviewed["validation_errors"] = reasons
    reviewed["valid_approval"] = reviewed["review_status"].eq("APPROVED") & reviewed["validation_errors"].eq("")
    return reviewed


def coverage_summary(src_path: Path, month_path: Path) -> pd.DataFrame:
    src = pd.read_csv(src_path, dtype={"source_id": "string"})
    src["source_type"] = src["source_type"].astype(str).str.lower().str.strip()
    src["source_id"] = src["source_id"].astype(str).str.strip()
    src = src.drop_duplicates(subset=["source_type", "source_id"])
    if not month_path.exists():
        src["monthly_data_status"] = "NOT_AVAILABLE"
        return src
    try:
        m = pd.read_csv(month_path, dtype={"source_id": "string"})
        if "source_type" not in m.columns:
            return src
        m["source_type"] = m["source_type"].astype(str).str.lower().str.strip()
        m["source_id"] = m["source_id"].astype(str).str.strip()
        count = m.groupby(["source_type", "source_id"]).size().rename("monthly_rows").reset_index()
        src = src.merge(count, on=["source_type", "source_id"], how="left")
        src["monthly_rows"] = src["monthly_rows"].fillna(0).astype(int)
        src["monthly_data_status"] = src["monthly_rows"].gt(0).map({True: "HAS_RECORDS", False: "NO_RECORDS"})
        return src
    except Exception as exc:
        src["monthly_data_status"] = f"COULD_NOT_PARSE: {exc}"
        return src


def main() -> None:
    p = argparse.ArgumentParser(description="Hydrological connectivity evidence review: no auto-approvals.")
    p.add_argument("--root", default=str(project_root()), help="Project root")
    p.add_argument("--input-dir", default="processed_data/multivariate_flood_modeling/06c_hydrological_spatial_mapping")
    p.add_argument("--output-dir", default="processed_data/multivariate_flood_modeling/06d_hydrological_connectivity")
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    cdir = root / args.input_dir
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    candidates = assemble_candidates(cdir)
    template = out / "01_candidate_review_template.csv"
    make_review_file(template, candidates)

    reviewed = pd.read_csv(template, dtype={"source_id": "string", "adm_id": "string"}, keep_default_na=False)
    evaluated = evaluate_review(reviewed, candidates)

    approved = evaluated[evaluated["valid_approval"]].copy()
    pending = evaluated[(evaluated["review_status"] == "PENDING") & evaluated["validation_errors"].eq("")].copy()
    rejected_invalid = evaluated[
        (evaluated["review_status"] == "REJECTED") | evaluated["validation_errors"].ne("")
    ].copy()

    approved[MAPPING_COLUMNS].to_csv(out / "02_reviewed_mapping_APPROVED.csv", index=False)
    pending.to_csv(out / "03_pending_review.csv", index=False)
    rejected_invalid.to_csv(out / "04_rejected_or_invalid.csv", index=False)
    coverage = coverage_summary(
        cdir / "01_hydrology_source_locations.csv",
        root / "processed_data/multivariate_flood_modeling/06a_lake_dfo_monthly/station_lake_monthly.csv",
    )
    coverage.to_csv(out / "05_source_coverage.csv", index=False)

    summary = {
        "candidate_count": int(len(candidates)),
        "reviewed_rows": int(len(evaluated)),
        "approved_with_evidence": int(len(approved)),
        "pending": int(len(pending)),
        "rejected_or_invalid": int(len(rejected_invalid)),
        "explicit_mapping_is_required": True,
        "geographic_proximity_is_not_hydrological_evidence": True,
        "notes": [
            "No candidate is approved automatically.",
            "An approved record is a documented human review, NOT automatic proof of connectivity.",
            "Review must use credible river network/catchment/flow direction evidence.",
            "Time lag and effective prediction-time availability need additional checks in the model.",
            "The 06b script will not automatically read this mapping. The reviewed mapping must first be used by 06a to construct ADM-level hydrology features.",
        ],
    }
    (out / "06_validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "07_review_instructions.md").write_text(
        """# 06d – Hydrological connectivity review

1. Open **01_candidate_review_template.csv**. It is only created if absent.
2. For each candidate with verified river-network/catchment evidence:
   - `review_status`: set `APPROVED`
   - `relationship`: one of `UPSTREAM_INFLOW`, `DOWNSTREAM_REACH`,
     `SAME_RIVER_REACH`, `CONNECTED_LAKE_OUTFLOW`, `SHARED_CATCHMENT`,
     `OTHER_DOCUMENTED`
   - `evidence_source`: a specific document, verified map, or link
   - `reviewer_name`: name or reviewer identifier
   - `review_date`: YYYY-MM-DD
   - `hydrological_notes`: river, travel direction, relevance, caveats
3. Set `REJECTED` for disproved links, leave `PENDING` otherwise.
4. Re-run 06d to generate **02_reviewed_mapping_APPROVED.csv**.
5. Import only approved rows into the ADM hydrology panel builder. The current
   06b driver-comparison code does NOT consume 06d CSV automatically.

**Important:** 06c geography only finds candidates; it does not prove
connectivity. Lake Victoria's downstream influence may require candidate links
outside the geographic-distance search. Do not fabricate those links. If you
have an authoritative network-based mapping, extend 06c's candidate table with
its documented source and review before approving.

An `APPROVED` row is evidence-recorded review, not an independently verified
hydrological result. A credible upstream relation and forecast-available lag
must be documented before training predictive models.
""", encoding="utf-8",
    )
    print(f"06d output: {out}")
    print(f"Geographic candidates: {len(candidates)}")
    print(f"Approved with documentation: {len(approved)}")
    print(f"Pending: {len(pending)}")
    print(f"Rejected/invalid: {len(rejected_invalid)}")
    if approved.empty:
        print("No verified hydrological links yet. Do NOT rerun 06b expecting lake/DFO inputs.")


if __name__ == "__main__":
    main()
