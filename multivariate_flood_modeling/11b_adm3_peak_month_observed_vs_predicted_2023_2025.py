#!/usr/bin/env python3
"""11b — ADM3 observed vs predicted peak month, 2023–2025.

A standalone POSTPROCESSING script: reads 10b monthly backtest outputs.
Does NOT retrain any model. Default: M4 + C combined + Monthly Rolling.
For each ADM3 and each year, choose the calendar month with the highest
recorded flood-event-date PROXY among 12 monthly actual values, and choose
independently the month of the highest among 12 month-ahead predictions.

IMPORTANT: this is RETROSPECTIVE identification of the peak among 12 separately
issued one-month-ahead forecasts. It does NOT mean the model forecast the
whole year's peak month at the beginning of the year.

Data quality:
- 12 valid paired months required by default. Missing regions remain grey.
- All zero/non-positive actual recorded activity -> 'No positive activity'.
- Maximum predicted activity < --minimum-peak (default 1) -> same class.
- Near-ties within --tie-tolerance (default 0.25 proxy dates) are ambiguous.
- Missing remote sensing detection does not prove absence of flooding.

Expected location:
  scripts/multivariate_flood_modeling/11b_adm3_peak_month_observed_vs_predicted_2023_2025.py
Run (from repo root or VS Code):
  python scripts/multivariate_flood_modeling/11b_adm3_peak_month_observed_vs_predicted_2023_2025.py

Outputs:
  processed_data/multivariate_flood_modeling/11b_adm3_peak_month_comparison/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

YEARS = (2023, 2024, 2025)
MODELS = {
    'M0_history': 'M0 = Flood history only',
    'M1_rainfall': 'M1 = Flood history + Rainfall',
    'M2_runoff': 'M2 = Flood history + Runoff',
    'M3_et0': 'M3 = Flood history + ET0',
    'M4_all_drivers': 'M4 = Flood history + Rainfall + Runoff + ET0',
}
MEMORIES = {
    'A_same_month_previous_years': 'A = same calendar month 1 & 2 years earlier',
    'B_recent_months': 'B = preceding 1, 2 & 3 months',
    'C_combined': 'C = previous 1–3 months + same month 1 & 2 years earlier',
}
# Bright seasonally ordered colours inspired by 10e map supplied by user.
MONTH_COLORS = {
    1: '#75C7F3', 2: '#63ADEC', 3: '#6599DE',
    4: '#88DCD4', 5: '#69BDB0', 6: '#54A69B',
    7: '#D7B6EE', 8: '#B98BDA', 9: '#8C66CF',
    10: '#BC59A5', 11: '#993B98', 12: '#7546B7',
}
SPECIAL_COLORS = {
    'NO_POSITIVE_RECORDED_ACTIVITY': '#E8E5EB',
    'TIED_OR_INDISTINCT_PEAK': '#EACCA3',
    'INSUFFICIENT_PAIRED_MONTHS': '#B8B3BF',
    'NO_MODEL_PREDICTION': '#B8B3BF',
}
OUTLINE = '#CBC6D3'


def project_root() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name == 'multivariate_flood_modeling' and p.parent.parent.name == 'scripts':
        return p.parents[2]
    return Path.cwd()


def load_boundary(root: Path, override: Path | None) -> gpd.GeoDataFrame:
    path = override if override else root / 'raw_data/administrative_boundaries/ssd_admin3.geojson'
    if not path.is_file():
        raise FileNotFoundError(f'ADM3 boundary file not found: {path}')
    geo = gpd.read_file(path)
    if geo.crs is None:
        raise ValueError(f'Boundary has no CRS: {path}')
    id_candidates = ('adm_id', 'adm3_pcode', 'ADM3_PCODE', 'admin3Pcod', 'ADM3_CODE', 'ADM3_PCode')
    id_col = next((c for c in id_candidates if c in geo.columns), None)
    if id_col is None:
        raise ValueError(f'ADM3 ID column not found in {path}. Columns: {list(geo.columns)}')
    geo['adm_id'] = geo[id_col].astype('string').str.strip()
    if geo.adm_id.isna().any() or geo.adm_id.duplicated().any():
        raise ValueError('ADM3 polygons contain null or duplicate administrative IDs')
    return geo[['adm_id', 'geometry']].to_crs('EPSG:6933')


def load_predictions(path: Path, model: str, memory: str, strategy: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f'10b prediction CSV not found: {path}. Run 10b first.')
    columns = ['level', 'memory', 'model', 'strategy', 'adm_id', 'year', 'month',
               'actual_event_dates', 'predicted_event_dates']
    df = pd.read_csv(path, usecols=columns, dtype={'adm_id': 'string'})
    df = df.loc[(df.level == 'ADM3') & (df.model == model) &
                (df.memory == memory) & (df.strategy == strategy) & df.year.isin(YEARS)].copy()
    if df.empty:
        raise ValueError('No matching ADM3 records; verify --model, --memory and --strategy')
    df['adm_id'] = df.adm_id.str.strip()
    if df[['actual_event_dates', 'predicted_event_dates']].isna().any().any():
        raise ValueError('Prediction CSV has null actual/predicted values')
    if df.duplicated(['adm_id', 'year', 'month']).any():
        raise ValueError('Duplicate ADM3/year/month predictions found')
    if not df.month.between(1, 12).all():
        raise ValueError('Invalid month values found')
    if (df[['actual_event_dates', 'predicted_event_dates']] < 0).any().any():
        raise ValueError('Negative target/prediction values found')
    return df


def classify_peak(d: pd.DataFrame, field: str, min_months: int,
                  minimum_peak: float, tie_tolerance: float) -> tuple[str, int | None, float]:
    values = d[['month', field]].dropna().copy()
    if values.month.nunique() < min_months:
        return 'INSUFFICIENT_PAIRED_MONTHS', None, float('nan')
    top = float(values[field].max())
    if top < minimum_peak:
        return 'NO_POSITIVE_RECORDED_ACTIVITY', None, top
    near = values.loc[values[field] >= top - tie_tolerance]
    if near.month.nunique() != 1:
        return 'TIED_OR_INDISTINCT_PEAK', None, top
    return 'PEAK_IDENTIFIED', int(near.month.iloc[0]), top


def annual_peak_table(df: pd.DataFrame, boundaries: gpd.GeoDataFrame,
                      min_months: int, actual_min: float,
                      predicted_min: float, tolerance: float) -> pd.DataFrame:
    # Start from all polygons so no-prediction regions remain visible, not dropped.
    valid_ids = set(boundaries.adm_id.astype(str))
    unknown = sorted(set(df.adm_id.astype(str)) - valid_ids)
    if unknown:
        raise ValueError(f'Predictions have {len(unknown)} ADM3 IDs not found in boundary (examples: {unknown[:8]})')
    pieces = []
    for year in YEARS:
        for adm_id in boundaries.adm_id.astype(str):
            group = df.loc[(df.adm_id == adm_id) & (df.year == year)]
            count = int(group.month.nunique())
            if count == 0:
                o_status = p_status = 'NO_MODEL_PREDICTION'
                om = pm = None
                ov = pv = float('nan')
            else:
                o_status, om, ov = classify_peak(group, 'actual_event_dates', min_months, actual_min, tolerance)
                p_status, pm, pv = classify_peak(group, 'predicted_event_dates', min_months, predicted_min, tolerance)
            if om is not None and pm is not None:
                distance = min(abs(om-pm), 12-abs(om-pm))
                agreement = 'MATCH' if distance == 0 else ('OFF_BY_1_MONTH' if distance == 1 else 'OFF_BY_2PLUS_MONTHS')
            else:
                distance = np.nan
                agreement = 'NOT_COMPARABLE'
            pieces.append(dict(
                adm_id=adm_id, year=year, paired_months=count,
                observed_peak_status=o_status, observed_peak_month=om, observed_peak_value=ov,
                predicted_peak_status=p_status, predicted_peak_month=pm, predicted_peak_value=pv,
                peak_month_circular_distance=distance, peak_month_agreement=agreement,
            ))
    return pd.DataFrame(pieces)


def color_for(row, kind: str) -> str:
    month = row[f'{kind}_peak_month']
    if pd.notna(month):
        return MONTH_COLORS[int(month)]
    return SPECIAL_COLORS[row[f'{kind}_peak_status']]


def plot_one(ax, geo: gpd.GeoDataFrame, kind: str, label: str):
    g = geo.copy()
    g['fill'] = g.apply(lambda r: color_for(r, kind), axis=1)
    # Plot by colour for efficient handling of hundreds of ADM3 polygons.
    for color, part in g.groupby('fill', sort=False):
        part.plot(ax=ax, color=color, edgecolor=OUTLINE, linewidth=.27, zorder=2)
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_title(label, fontsize=19, fontweight='bold', pad=14, color='#28242C')


def save_figure(fig, out: Path, stem: str):
    for extension in ('png', 'pdf'):
        fig.savefig(out / f'{stem}.{extension}', dpi=310, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def make_year_figure(year: int, boundary: gpd.GeoDataFrame, audit: pd.DataFrame,
                     model: str, memory: str, strategy: str, out: Path):
    data = audit.loc[audit.year == year].drop(columns='year')
    geo = boundary.merge(data, on='adm_id', how='left', validate='one_to_one')
    if geo.observed_peak_status.isna().any():
        raise ValueError('Not all boundary ADM3 IDs were covered in the annual audit')
    fig = plt.figure(figsize=(22, 11.7), facecolor='white')
    ax_left = fig.add_axes([.016, .17, .37, .72])
    ax_right = fig.add_axes([.395, .17, .37, .72])
    leg = fig.add_axes([.795, .18, .195, .67]); leg.axis('off')
    plot_one(ax_left, geo, 'observed', f'{year} — Observed peak month')
    plot_one(ax_right, geo, 'predicted', f'{year} — Predicted peak month')
    legend = [Patch(color=MONTH_COLORS[m], label=f'{m:02d} — {pd.Timestamp(2024,m,1).strftime("%B")}') for m in range(1,13)]
    legend += [
        Patch(color=SPECIAL_COLORS['NO_POSITIVE_RECORDED_ACTIVITY'], label='No positive / insufficient signal'),
        Patch(color=SPECIAL_COLORS['TIED_OR_INDISTINCT_PEAK'], label='Tied / indistinct peak'),
        Patch(color=SPECIAL_COLORS['NO_MODEL_PREDICTION'], label='No comparable prediction / <12 months'),
    ]
    leg.legend(handles=legend, loc='center left', title='Calendar month / special status',
               frameon=False, fontsize=10.2, title_fontsize=11.6,
               labelspacing=.54, borderaxespad=0, handlelength=1.7)
    fig.text(.5, .97, f'ADM3 | Observed vs Predicted Peak Month | {year}',
             ha='center', va='top', fontsize=23, fontweight='bold', color='#29262D')
    nall = len(geo)
    npred = int(geo.paired_months.eq(12).sum())
    both = geo.loc[geo.peak_month_agreement.ne('NOT_COMPARABLE')]
    correct = int(both.peak_month_agreement.eq('MATCH').sum())
    note = f'{npred} / {nall} ADM3 have all 12 paired monthly predictions. Peak-month agreement: {correct}/{len(both)} among areas with identifiable peaks.'
    fig.text(.5, .126, note, ha='center', fontsize=11, weight='bold', color='#3F3549')
    fig.text(.5, .091, f'{MODELS[model]}  |  {MEMORIES[memory]}  |  {strategy}',
             ha='center', fontsize=10.1, color='#3F3549')
    fig.text(.5, .064, 'Each peak is identified retrospectively from 12 separate next-month predictions; NOT a forecast of the full year made in January.',
             ha='center', fontsize=9.2, color='#554B5E')
    fig.text(.5, .039, 'Recorded flood-event-date PROXY, not verified inundation duration. Missing detection is not proof of no flooding.',
             ha='center', fontsize=9.0, color='#554B5E')
    save_figure(fig, out, f'01_adm3_peak_month_{year}_observed_vs_predicted')


def make_agreement_chart(audit: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(11, 5.3))
    fig.subplots_adjust(left=.12, right=.97, top=.83, bottom=.19)
    groups = ['MATCH','OFF_BY_1_MONTH','OFF_BY_2PLUS_MONTHS','NOT_COMPARABLE']
    labels = ['Same month', '±1 month', 'Difference ≥2 months', 'Not comparable']
    colors = ['#54A69B', '#7EC9F2', '#7546B7', '#C5C1CA']
    x = np.arange(len(YEARS)); bottom = np.zeros(len(YEARS), dtype=int)
    for cat, label, clr in zip(groups, labels, colors):
        values = np.array([int(audit.loc[audit.year.eq(y), 'peak_month_agreement'].eq(cat).sum()) for y in YEARS])
        bars = ax.bar(x, values, bottom=bottom, color=clr, label=label, width=.58)
        for bar, value, base in zip(bars, values, bottom):
            if value >= 10:
                ax.text(bar.get_x()+bar.get_width()/2, base+value/2, f'{value}', ha='center', va='center', fontsize=10, weight='bold')
        bottom += values
    ax.set_xticks(x, [str(y) for y in YEARS])
    ax.set_ylabel('Number of ADM3 areas')
    ax.set_title('Annual observed–predicted peak-month agreement (all ADM3)', fontsize=14, weight='bold')
    ax.legend(loc='upper right', frameon=False, ncol=2)
    ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    save_figure(fig, out, '02_peak_month_agreement_summary')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    root = project_root()
    ap.add_argument('--root', type=Path, default=root)
    ap.add_argument('--predictions', type=Path, default=None, help='10b 01_test_predictions_monthly.csv')
    ap.add_argument('--boundary', type=Path, default=None, help='ADM3 boundary GeoJSON')
    ap.add_argument('--output', type=Path, default=None)
    ap.add_argument('--model', choices=MODELS, default='M4_all_drivers')
    ap.add_argument('--memory', choices=MEMORIES, default='C_combined')
    ap.add_argument('--strategy', choices=['Fixed','Monthly Rolling'], default='Monthly Rolling')
    ap.add_argument('--min-months', type=int, default=12, help='Required paired months per year; default 12')
    ap.add_argument('--minimum-observed-peak', type=float, default=0.000001,
                    help='Smallest positive observed activity to identify a peak')
    ap.add_argument('--minimum-predicted-peak', type=float, default=1.0,
                    help='Predictions below this maximum are not assigned a meaningful peak')
    ap.add_argument('--tie-tolerance', type=float, default=.25,
                    help='If two peak candidates differ by ≤ this number of proxy event dates, mark indistinct')
    args = ap.parse_args()
    if not 1 <= args.min_months <= 12:
        ap.error('--min-months must be between 1 and 12')
    if args.minimum_observed_peak < 0 or args.minimum_predicted_peak < 0 or args.tie_tolerance < 0:
        ap.error('minimum peak and tie tolerance must not be negative')
    root = args.root.resolve()
    path = args.predictions or root / 'processed_data/multivariate_flood_modeling/10b_monthly_rolling_origin/01_test_predictions_monthly.csv'
    out = args.output or root / 'processed_data/multivariate_flood_modeling/11b_adm3_peak_month_comparison'
    out.mkdir(parents=True, exist_ok=True)
    boundary = load_boundary(root, args.boundary)
    predictions = load_predictions(path, args.model, args.memory, args.strategy)
    audit = annual_peak_table(predictions, boundary, args.min_months,
                              args.minimum_observed_peak, args.minimum_predicted_peak,
                              args.tie_tolerance)
    audit.to_csv(out / '00_adm3_year_peak_month_audit.csv', index=False)
    coverage = []
    for year in YEARS:
        sub = audit[audit.year == year]
        coverage.append({
            'year': year, 'all_adm3': len(sub),
            'full_12_month_predictions': int(sub.paired_months.eq(12).sum()),
            'no_predictions': int(sub.paired_months.eq(0).sum()),
            'partial_predictions': int(sub.paired_months.between(1,11).sum()),
            'identifiable_observed_peaks': int(sub.observed_peak_month.notna().sum()),
            'identifiable_predicted_peaks': int(sub.predicted_peak_month.notna().sum()),
            'both_peaks_identifiable': int(sub.peak_month_agreement.ne('NOT_COMPARABLE').sum()),
            'exact_agreement': int(sub.peak_month_agreement.eq('MATCH').sum()),
            'within_one_month': int(sub.peak_month_agreement.isin(['MATCH','OFF_BY_1_MONTH']).sum()),
        })
        make_year_figure(year, boundary, audit, args.model, args.memory, args.strategy, out)
    pd.DataFrame(coverage).to_csv(out / '03_annual_coverage_and_agreement.csv', index=False)
    make_agreement_chart(audit, out)
    metadata = {
        'data': str(path), 'boundary': str(args.boundary or root / 'raw_data/administrative_boundaries/ssd_admin3.geojson'),
        'model': args.model, 'memory': args.memory, 'strategy': args.strategy,
        'years': list(YEARS), 'minimum_paired_months': args.min_months,
        'minimum_observed_peak': args.minimum_observed_peak,
        'minimum_predicted_peak': args.minimum_predicted_peak,
        'tie_tolerance': args.tie_tolerance,
        'peak_definition': 'calendar month with maximum actual or forecast recorded event-date proxy within each year',
        'important_limitations': [
            'Retrospective yearly peak selected from separate next-month forecasts, not a January full-year forecast.',
            'No historical flood observations outside model cohort are imputed into observed comparison.',
            'Observed zero is zero records in proxy, not verified absence of inundation.',
            'Predicted nonzero values below the configurable minimum are not assigned a peak.',
            'No flood-occurrence probabilities or confirmed daily flood duration are estimated.',
            'Test-period results may have influenced earlier model choices during project development.',
        ],
    }
    (out / '04_methodology.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Output: {out}')
    print(pd.DataFrame(coverage).to_string(index=False))
    print('Note: Peak-month agreement is retrospective and proxy-based, not operational warning validation.')


if __name__ == '__main__':
    main()
