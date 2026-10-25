#!/usr/bin/env python3
"""13c — South Sudan ADM3 historical flood activity × exposure screening.

Run (no arguments required):
  python scripts/multivariate_flood_modeling/13c_analyze_flood_exposure.py

Optional --flood-zip, --impact-zip and --out overrides are supported.

The script outputs descriptive area-level screening, NOT observed inundated
population, damaged farmland, flooded facilities or a prospective forecast.
"""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import zipfile
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FLOOD_CSV = '02_adm3_period_summary.csv'
IMPACT_CSV = '01_adm3_impact_indicators.csv'
QUALITY_CSV = '06_data_quality_flags.csv'


def load_csv(path: Path, filename: str) -> pd.DataFrame:
    """Read a CSV from a ZIP, an extracted directory, or a direct CSV file."""
    if path.is_file() and path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as z:
            options = [name for name in z.namelist() if name.endswith('/' + filename) or name == filename]
            options = [n for n in options if not n.startswith('__MACOSX/')]
            if len(options) != 1:
                raise ValueError(f'Expected exactly one {filename} in {path}; found {options}')
            return pd.read_csv(io.BytesIO(z.read(options[0])), dtype={'adm_id': 'string'})
    if path.is_dir():
        matches = [p for p in path.rglob(filename) if '__MACOSX' not in p.parts]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one {filename} under {path}; found {matches}')
        return pd.read_csv(matches[0], dtype={'adm_id':'string'})
    if path.name == filename:
        return pd.read_csv(path, dtype={'adm_id':'string'})
    raise FileNotFoundError(f'Cannot locate {filename} in {path}')


def clean_ids(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if 'adm_id' not in df.columns:
        raise ValueError(f'{source}: no adm_id column')
    df = df.copy()
    df['adm_id'] = df['adm_id'].astype('string').str.strip().str.upper()
    if df['adm_id'].isna().any() or (df['adm_id'] == '').any() or df['adm_id'].duplicated().any():
        raise ValueError(f'{source}: blank or duplicated adm_id detected')
    return df


def robust_rank(series: pd.Series) -> pd.Series:
    """Percentile within the observed comparison set, for descriptive plots only."""
    return series.rank(pct=True, method='average')


def scatter_plot(df: pd.DataFrame, x: str, y: str, title: str, filename: Path) -> None:
    valid = df[[x,y]].replace([np.inf,-np.inf],np.nan).dropna()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(valid[x],valid[y],alpha=.55,s=18)
    ax.set(title=title,xlabel=x.replace('_',' '),ylabel=y.replace('_',' '))
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(filename,dpi=180)
    plt.close(fig)


def main() -> None:
    # Anchor default paths to the repository rather than the terminal working directory.
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / 'processed_data' / 'multivariate_flood_modeling'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--flood-zip', type=Path,
                        default=data_dir / '08b_before_after_2019.zip',
                        help='08b ZIP or extracted directory')
    parser.add_argument('--impact-zip', type=Path,
                        default=data_dir / '13b_adm3_impact_integration.zip',
                        help='13b ZIP or extracted directory')
    parser.add_argument('--out', type=Path,
                        default=data_dir / '13c_flood_exposure_screening',
                        help='Output directory')
    args = parser.parse_args()
    for label, path in [('08b', args.flood_zip), ('13b', args.impact_zip)]:
        if not path.exists():
            raise FileNotFoundError(f'{label} input not found: {path}')
    out = args.out
    out.mkdir(parents=True,exist_ok=True)
    flood=clean_ids(load_csv(args.flood_zip,FLOOD_CSV),'08b')
    impact=clean_ids(load_csv(args.impact_zip,IMPACT_CSV),'13b')
    quality=clean_ids(load_csv(args.impact_zip,QUALITY_CSV),'13b quality')
    expected_flood=['region_name','parent_adm2_name','mean_2019-2025','mean_2000-2010','mean_2011-2018','change_after_2019']
    expected_impact=['adm3_name','adm2_name','population_estimate_2025','population_data_fraction_2025','health_facilities_total','cropland_area_km2','rangeland_area_km2']
    for name,data,cols in [('08b',flood,expected_flood),('13b',impact,expected_impact)]:
        absent=[c for c in cols if c not in data]
        if absent:raise ValueError(f'{name} missing required columns: {absent}')
    flood_ids=set(flood.adm_id)
    impact_ids=set(impact.adm_id)
    if flood_ids != impact_ids:
        raise ValueError(f'ADM3 ID mismatches: only 08b={sorted(flood_ids-impact_ids)[:15]}, only 13b={sorted(impact_ids-flood_ids)[:15]}')
    if set(quality.adm_id) != flood_ids:
        raise ValueError('Quality flag ID set differs from 08b/13b')
    merged=flood.merge(impact,on='adm_id',how='inner',validate='one_to_one').merge(quality.drop(columns=['adm3_name']),on='adm_id',validate='one_to_one')
    assert len(merged)==len(flood)==len(impact)
    merged['name_match']=merged.region_name.astype('string').str.casefold().str.strip().eq(merged.adm3_name.astype('string').str.casefold().str.strip())
    merged['parent_match']=merged.parent_adm2_name.astype('string').str.casefold().str.strip().eq(merged.adm2_name.astype('string').str.casefold().str.strip())
    if not merged.name_match.all() or not merged.parent_match.all():
        raise ValueError('ADM names or parent ADM2 do not match despite matching IDs; investigate before analysis')
    merged=merged.rename(columns={'mean_2019-2025':'historical_flood_days_mean_2019_2025','change_after_2019':'historical_flood_days_change_after_2019'})
    columns=['historical_flood_days_mean_2019_2025','historical_flood_days_change_after_2019','health_facilities_total','cropland_area_km2','rangeland_area_km2','population_estimate_2025','population_data_fraction_2025']
    for col in columns: merged[col]=pd.to_numeric(merged[col],errors='coerce')
    # Crucial: a small WorldPop valid-pixel fraction is not a population exposure estimate.
    # By default, we do not treat incomplete coverage as a trustworthy complete population total.
    merged['population_full_coverage_flag']=(merged['population_data_fraction_2025']>=.99)&merged['population_estimate_2025'].notna()
    merged['population_use_for_priority']=False
    # Distinguish absence of facilities (0) from lack of information (NaN).
    merged['flood_high_relative']=merged['historical_flood_days_mean_2019_2025']>=merged['historical_flood_days_mean_2019_2025'].quantile(.75)
    merged['flood_increase_high_relative']=merged['historical_flood_days_change_after_2019']>=merged['historical_flood_days_change_after_2019'].quantile(.75)
    for c in ['health_facilities_total','cropland_area_km2','rangeland_area_km2']:
        merged[c+'_high_relative']=merged[c]>=merged[c].quantile(.75)
    merged['overlap_high_hazard_facilities']=merged['flood_high_relative']&merged['health_facilities_total_high_relative']
    merged['overlap_high_hazard_cropland']=merged['flood_high_relative']&merged['cropland_area_km2_high_relative']
    merged['overlap_high_hazard_rangeland']=merged['flood_high_relative']&merged['rangeland_area_km2_high_relative']
    merged['high_hazard_exposure_dimensions_count']=merged[['overlap_high_hazard_facilities','overlap_high_hazard_cropland','overlap_high_hazard_rangeland']].sum(axis=1)
    # Not a final humanitarian priority score or verified impact indicator.
    merged['exploratory_overlap_only']=True
    merged=merged.sort_values(['historical_flood_days_mean_2019_2025','adm_id'],ascending=[False,True])
    merged.to_csv(out/'01_adm3_flood_exposure_merged.csv',index=False)
    profile_cols=['adm_id','region_name','parent_adm2_name','historical_flood_days_mean_2019_2025','historical_flood_days_change_after_2019','health_facilities_total','cropland_area_km2','rangeland_area_km2','population_estimate_2025','population_data_fraction_2025','population_full_coverage_flag','overlap_high_hazard_facilities','overlap_high_hazard_cropland','overlap_high_hazard_rangeland','high_hazard_exposure_dimensions_count']
    merged.sort_values(['high_hazard_exposure_dimensions_count','historical_flood_days_mean_2019_2025'],ascending=[False,False])[profile_cols].to_csv(out/'02_exploratory_adm3_overlap_profiles.csv',index=False)
    metrics=['historical_flood_days_mean_2019_2025','historical_flood_days_change_after_2019','health_facilities_total','cropland_area_km2','rangeland_area_km2']
    corr=merged[metrics].corr(method='spearman')
    corr.to_csv(out/'03_spearman_correlations.csv',index=True)
    summary={'adm3_flood_count':len(flood),'adm3_impact_count':len(impact),'adm3_matched':len(merged),'name_matches':int(merged.name_match.sum()),'parent_matches':int(merged.parent_match.sum()),'missing_population_2025':int(merged.population_estimate_2025.isna().sum()),'population_coverage_below_99pct':int((merged.population_data_fraction_2025<.99).sum()),'median_population_data_fraction_2025':float(merged.population_data_fraction_2025.median()),'high_hazard_adm3_count':int(merged.flood_high_relative.sum()),'overlap_high_hazard_facilities_n':int(merged.overlap_high_hazard_facilities.sum()),'overlap_high_hazard_cropland_n':int(merged.overlap_high_hazard_cropland.sum()),'overlap_high_hazard_rangeland_n':int(merged.overlap_high_hazard_rangeland.sum()),'population_prioritization_enabled':False,'notes':['All exposure variables describe ADM3 area totals or counts, not measured flood exposure.','Flood metric is days with recorded flood activity in an administrative unit, NOT full-area inundation days.','Population coverage requires independent investigation; population withheld from priority/overlap classification.','Top-quartile overlaps are exploratory descriptive flags; not an operational humanitarian priority index.','Temporal mismatch exists: 2019–2025 flood history versus 2025 population snapshot and possibly undated static facility/land-use layers.','No projection, forward forecast, or flood footprint intersection is performed.']}
    (out/'04_quality_and_methodology.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    for c,short in [('health_facilities_total','health_facilities'),('cropland_area_km2','cropland'),('rangeland_area_km2','rangeland')]:
        scatter_plot(merged,'historical_flood_days_mean_2019_2025',c,f'ADM3 historical flood activity vs {short.replace("_"," ")}',out/f'05_flood_vs_{short}.png')
    print(json.dumps(summary,indent=2))
    print(f'Wrote outputs to: {out.resolve()}')

if __name__=='__main__':main()
