#!/usr/bin/env python3
"""Read-only 13f diagnostic collector. No downloads or third-party packages.

Reads the original monthly CSV and existing audit reports. It does not execute
05a/13f, interpret unknown statuses, fill missing values, or change source files.
"""
from pathlib import Path
from datetime import datetime
from collections import Counter
import argparse
import calendar
import csv
import hashlib
import json
import math
import sys
import zipfile

BASE = Path("processed_data/multivariate_flood_modeling")
MISSING = {"", "nan", "na", "n/a", "none", "null", "<na>"}


def project_root(explicit=None):
    starts = [Path(explicit).expanduser()] if explicit else [Path(__file__).resolve().parent, Path.cwd()]
    for start in starts:
        for folder in (start.resolve(), *start.resolve().parents):
            if (folder / BASE).is_dir():
                return folder
    raise FileNotFoundError("Project not found. Save under scripts/multivariate_flood_modeling or use --root PATH.")


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def monthly_summary(path):
    columns, rows = read_csv(path)
    lookup = {c.strip().lower(): c for c in columns}
    def field(*names):
        return next((lookup[n] for n in names if n in lookup), None)
    idcol = field("adm_id", "adm3_id", "adm3_pcode", "admin_id")
    yearcol, monthcol = field("year"), field("month")
    daycol = field("flood_days", "recorded_flood_days", "recorded_activity_days", "event_date_proxy")
    statuscol = field("target_status")
    result = {"path": str(path), "rows": len(rows), "columns": columns,
              "fields": {"id": idcol, "year": yearcol, "month": monthcol, "days": daycol, "status": statuscol}}
    if not all((idcol, yearcol, monthcol, daycol)):
        result["error"] = "Required columns were not found; original file is included for review."
        return result
    counts, statuses, years, keys = Counter(), Counter(), Counter(), Counter()
    for row in rows:
        statuses[str(row.get(statuscol, "") or "").strip() or "(blank)"] += 1
        aid = str(row.get(idcol, "") or "").strip().upper()
        if aid.casefold() in MISSING:
            counts["missing_id_rows"] += 1
        try:
            y, m = float(row[yearcol]), float(row[monthcol])
            if not (y.is_integer() and m.is_integer() and 1 <= y <= 9999 and 1 <= m <= 12):
                raise ValueError
            y, m = int(y), int(m)
        except (ValueError, TypeError, OverflowError):
            counts["invalid_year_month_rows"] += 1
            continue
        years[str(y)] += 1
        if aid.casefold() not in MISSING:
            keys[(aid, y, m)] += 1
        text = str(row.get(daycol, "") or "").strip()
        if text.casefold() in MISSING:
            counts["missing_day_values"] += 1
            continue
        try:
            days = float(text)
            if not math.isfinite(days) or not days.is_integer() or not 0 <= days <= calendar.monthrange(y, m)[1]:
                raise ValueError
        except (ValueError, TypeError, OverflowError):
            counts["invalid_day_values"] += 1
            continue
        counts["zero_day_values" if days == 0 else "positive_day_values"] += 1
    result.update(counts=dict(counts), target_status_counts=dict(statuses), rows_by_year=dict(sorted(years.items())),
                  duplicate_id_year_month_groups=sum(n > 1 for n in keys.values()),
                  note="Raw record counts only. Zero values and statuses are NOT validated observation coverage.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="Optional project root")
    args = parser.parse_args()
    root = project_root(args.root)
    base = root / BASE
    audit = base / "13f_top10_adm3_profiles/monthly_audit"
    scripts = root / "scripts/multivariate_flood_modeling"
    monthly = base / "05a_monthly_panel/adm3_monthly_panel.csv"
    paths = [monthly] if monthly.is_file() else []
    if audit.is_dir():
        paths += [p for p in audit.iterdir() if p.is_file() and p.suffix.lower() in {".csv", ".json"}]
    for pattern in ("05a_build_monthly_panel*.py", "13f_map_top10_adm3_profiles*.py"):
        paths += list(scripts.glob(pattern))
    paths += list((base / "13f_top10_adm3_profiles").glob("0[12]_top10_*.csv"))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"No diagnostic inputs at the expected paths under {base}")
    print(f"[ROOT] {root}", flush=True)
    before = {p: file_hash(p) for p in paths}
    report = {"purpose": "Read-only input and audit collection; no data repair or new peak-month estimates.",
              "monthly_csv_found": monthly.is_file(), "audit_directory_found": audit.is_dir(),
              "source_code_found": [p.name for p in paths if p.suffix == ".py"]}
    if monthly.is_file():
        print("[READ] Original monthly CSV", flush=True)
        report["raw_monthly_summary"] = monthly_summary(monthly)
    for name in ("02_status_inventory.csv", "05_profile_coverage.csv"):
        path = audit / name
        if not path.is_file():
            report[name] = {"found": False}
            continue
        _, rows = read_csv(path)
        report[name] = {"rows": len(rows), "first_20_rows": rows[:20]}
        if name.startswith("05_"):
            report["profile_status_counts"] = dict(Counter(r.get("profile_status", "(absent)") for r in rows))
    quality = audit / "04_monthly_quality.json"
    if quality.is_file():
        try:
            old = json.loads(quality.read_text(encoding="utf-8-sig"))
            report["previous_quality_report"] = old
            previous_hash = old.get("source_csv_sha256")
            report["audit_matches_current_monthly_csv"] = (previous_hash == before[monthly]
                if previous_hash and monthly in before else None)
        except (ValueError, AttributeError) as exc:
            report["quality_json_read_error"] = str(exc)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = base / "13f_monthly_diagnostics" / f"run_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    zipped = out.parent / f"13f_monthly_check_{stamp}.zip"
    with zipfile.ZipFile(zipped, "w", zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            relative = path.relative_to(root).as_posix()
            print(f"[COPY TO ZIP] {relative}", flush=True)
            z.write(path, "inputs/" + relative)
        changed = [str(p.relative_to(root)) for p in paths if file_hash(p) != before[p]]
        report["inputs_unchanged_during_collection"] = not changed
        report["changed_input_paths"] = changed
        report["input_manifest"] = [{"path": str(p.relative_to(root)), "sha256_before": before[p]} for p in paths]
        target = out / "00_diagnostic_summary.json"
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        z.write(target, target.name)
    if changed:
        print("[WARNING] Inputs changed during collection. Stop other writers and rerun; do not rely on this bundle.")
        return 1
    print("[OK] Original input contents are unchanged.", flush=True)
    if not monthly.is_file():
        print(f"[WARNING] Monthly CSV not found at: {monthly}")
    if report.get("audit_matches_current_monthly_csv") is False:
        print("[WARNING] Existing audit refers to a different version of the monthly CSV.")
    print("[PROFILE STATUS] " + json.dumps(report.get("profile_status_counts", {}), ensure_ascii=False))
    print(f"[SUMMARY] {target}", flush=True)
    print(f"[SHARE ZIP] {zipped}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, csv.Error, zipfile.BadZipFile) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
