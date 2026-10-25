#!/usr/bin/env python3
"""JBG060: annotate existing ADM3 ERA5 annual output without filling missing days.

Read-only with respect to Script 04 outputs. Writes a separate annotated CSV and QC
summary. PASS remains Script 04's original coverage-based status.
"""
from pathlib import Path
import argparse
import calendar
import pandas as pd
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = p.parse_args()
    folder = args.root / 'processed_data/multivariate_flood_modeling/04_adm3_rainfall_runoff'
    source = folder / 'adm3_annual_rainfall_runoff.csv'
    if not source.is_file():
        p.error(f'Cannot find Script 04 output: {source}')
    df = pd.read_csv(source)
    required = {'adm_id', 'year', 'rainfall_valid_days', 'runoff_valid_days', 'hydrology_qc_status'}
    if not required.issubset(df.columns):
        p.error(f'Missing required columns: {sorted(required - set(df.columns))}')
    if df.duplicated(['adm_id', 'year']).any():
        p.error('Duplicate ADM3-year keys; stop and investigate')
    for col in ('year', 'rainfall_valid_days', 'runoff_valid_days'):
        df[col] = pd.to_numeric(df[col], errors='raise')
    df['days_expected'] = df['year'].astype(int).map(lambda y: 366 if calendar.isleap(y) else 365)
    for role in ('rainfall', 'runoff'):
        n = df[f'{role}_valid_days']
        if ((n < 0) | (n > df['days_expected'])).any():
            p.error(f'Invalid {role} day counts')
        df[f'{role}_missing_days'] = df['days_expected'] - n
    both_full = ((df['rainfall_missing_days'] == 0) & (df['runoff_missing_days'] == 0))
    accepted = df['hydrology_qc_status'].eq('PASS')
    df['hydrology_completeness'] = np.select(
        [accepted & both_full, accepted & ~both_full, ~accepted],
        ['COMPLETE_ALL_DAYS', 'ACCEPTED_PARTIAL_YEAR', 'INSUFFICIENT_COVERAGE'],
        default='REVIEW'
    )
    annotated = folder / 'adm3_annual_rainfall_runoff_with_completeness.csv'
    summary = folder / 'adm3_completeness_summary.csv'
    df.to_csv(annotated, index=False)
    (df.groupby(['year', 'hydrology_completeness'], dropna=False)
       .size().rename('region_count').reset_index().to_csv(summary, index=False))
    print('Original unchanged:', source)
    print('Annotated output:', annotated)
    print('QC summary:', summary)
    print('\n2025 breakdown:')
    print(df.loc[df.year.eq(2025), ['hydrology_completeness', 'rainfall_valid_days', 'runoff_valid_days']]
          .value_counts(dropna=False).to_string())


if __name__ == '__main__':
    main()
