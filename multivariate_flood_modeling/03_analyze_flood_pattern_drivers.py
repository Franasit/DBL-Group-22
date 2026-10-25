#!/usr/bin/env python3
"""JBG060 | Script 03: Flood patterns and drivers exploratory analysis.

Run from DBL-Group-22:
  python scripts/multivariate_flood_modeling/03_analyze_flood_pattern_drivers.py
  python scripts/multivariate_flood_modeling/03_analyze_flood_pattern_drivers.py --root /path/to/DBL-Group-22

Inputs: Script 01/02 full_analysis_panel.csv, Script 04 ADM3 hydrology CSV;
optional 05 completeness-labelled CSV and optional external, validated pattern file.
No model fitting, imputation, extrapolation, or modification of upstream files.
Correlations are descriptive, not causal effects or forecast evaluation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = 'processed_data/multivariate_flood_modeling'
DRIVERS = {
    'rainfall': ('era5_adm2_annual__annual_rainfall_mm', 'rainfall_annual_mm'),
    'runoff': ('era5_adm2_annual__annual_runoff_mm', 'runoff_annual_mm'),
    'et0': ('et0_adm2_annual__et0_annual_mm', 'et0_adm3_annual__et0_annual_mm'),
}
OUTCOMES = ['flood__flood_days', 'flood__annual_unique_flooded_area_km2',
            'flood__annual_flooded_ratio', 'flood__flood_event_count']


def load_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing input: {path}')
    return pd.read_csv(path, low_memory=False)


def validate_keys(df, label):
    if not {'adm_id','year'}.issubset(df.columns):
        raise ValueError(f'{label} missing adm_id/year')
    df = df.copy()
    df['adm_id'] = df['adm_id'].astype('string').str.strip()
    df['year'] = pd.to_numeric(df.year, errors='raise').astype(int)
    if df.adm_id.isna().any() or df.adm_id.eq('').any() or df.duplicated(['adm_id','year']).any():
        raise ValueError(f'{label}: missing IDs or repeated (adm_id,year) keys')
    return df.sort_values(['adm_id','year']).reset_index(drop=True)


def load_level(root, level):
    p = root / BASE / f'{"01_adm2" if level == "ADM2" else "02_adm3"}_panel_dataset/full_analysis_panel.csv'
    df = validate_keys(load_csv(p), str(p))
    if 'flood__flood_days' not in df.columns:
        raise ValueError(f'{p}: missing flood days target')
    if level == 'ADM3':
        folder = root / BASE / '04_adm3_rainfall_runoff'
        enhanced = folder / 'adm3_annual_rainfall_runoff_with_completeness.csv'
        source = enhanced if enhanced.exists() else folder / 'adm3_annual_rainfall_runoff.csv'
        if source.exists():
            hydro = validate_keys(load_csv(source), str(source))
            keep = ['adm_id','year'] + [c for c in ['rainfall_annual_mm','runoff_annual_mm',
                      'rainfall_valid_days','runoff_valid_days','hydrology_qc_status',
                      'hydrology_completeness','domain_coverage_fraction'] if c in hydro]
            df = df.merge(hydro[keep], on=['adm_id','year'], how='left', validate='one_to_one', indicator=True)
            df['adm3_hydro_joined'] = df.pop('_merge').eq('both')
        else:
            print('WARNING: ADM3 hydrology absent; M1/M2 cannot be evaluated for ADM3.', flush=True)
    out_names = {k: cols[0 if level == 'ADM2' else 1] for k, cols in DRIVERS.items()}
    for name, col in out_names.items():
        df[name] = pd.to_numeric(df[col], errors='coerce') if col in df else np.nan
    for col in OUTCOMES:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Honor original quality rules; never infer that PASS means all 365 days were observed.
    if level == 'ADM2' and 'qc__era5_suspect_zero_with_missing_daily' in df:
        bad = df['qc__era5_suspect_zero_with_missing_daily'].astype('string').str.lower().eq('true')
        df.loc[bad, ['rainfall','runoff']] = np.nan
    if level == 'ADM3' and 'hydrology_qc_status' in df:
        bad = df.hydrology_qc_status.astype('string').ne('PASS').fillna(True)
        df.loc[bad, ['rainfall','runoff']] = np.nan
    # Preserve partial years (including 2025), without incorrectly labelling them complete.
    if level == 'ADM3' and 'hydrology_completeness' not in df:
        if {'rainfall_valid_days','runoff_valid_days'}.issubset(df):
            expected = df.year.map(lambda y: 366 if pd.Timestamp(f'{int(y)}-12-31').is_leap_year else 365)
            full = df.rainfall_valid_days.eq(expected) & df.runoff_valid_days.eq(expected)
            accepted = df.hydrology_qc_status.astype('string').eq('PASS') if 'hydrology_qc_status' in df else pd.Series(False, index=df.index)
            df['hydrology_completeness'] = np.select([full & accepted, accepted],
                ['COMPLETE_ALL_DAYS','ACCEPTED_PARTIAL_YEAR'], default='INSUFFICIENT_COVERAGE')
    return df


def correlations(df, level):
    records = []
    for feature in ['rainfall','runoff','et0']:
        for outcome in OUTCOMES:
            if outcome not in df: continue
            # Same-year relations are explanatory/descriptive, NOT a next-year forecast.
            for mode in ['same_year','previous_year_to_current','within_admin_same_year']:
                temp = df[['adm_id','year',feature,outcome]].copy()
                if mode == 'previous_year_to_current':
                    prev = temp.groupby('adm_id')['year'].shift(1)
                    temp['x'] = temp.groupby('adm_id')[feature].shift(1).where(prev.eq(temp.year-1))
                    temp['y'] = temp[outcome]
                elif mode == 'within_admin_same_year':
                    temp['x'] = temp[feature] - temp.groupby('adm_id')[feature].transform('mean')
                    temp['y'] = temp[outcome] - temp.groupby('adm_id')[outcome].transform('mean')
                else:
                    temp['x'] = temp[feature]
                    temp['y'] = temp[outcome]
                tmp = temp[['x','y']].replace([np.inf,-np.inf],np.nan).dropna()
                ok = len(tmp) >= 10 and tmp.x.nunique() >= 2 and tmp.y.nunique() >= 2
                records.append({'level':level,'feature':feature,'outcome':outcome,'relationship':mode,
                                'n_valid_pairs':len(tmp),'n_unique_admin':int(temp.loc[tmp.index,'adm_id'].nunique()),
                                'pearson_r':float(tmp.x.corr(tmp.y)) if ok else np.nan,
                                'spearman_r':float(tmp.x.corr(tmp.y,method='spearman')) if ok else np.nan})
    return pd.DataFrame(records)


def coverage(df, level):
    out=[]
    for yr, block in df.groupby('year'):
        row={'level':level,'year':int(yr),'regions':int(block.adm_id.nunique()),
             'flood_days_valid':int(block['flood__flood_days'].notna().sum())}
        for name in ['rainfall','runoff','et0']:
            row[name+'_valid']=int(block[name].notna().sum())
        if 'hydrology_completeness' in block:
            row['partial_hydrology_regions']=int(block.hydrology_completeness.eq('ACCEPTED_PARTIAL_YEAR').sum())
        out.append(row)
    return pd.DataFrame(out)


def external_patterns(df, level, file_path, id_col, label_col):
    """Optional history-derived pattern labels for *descriptive* comparisons only.
    This MUST NOT be used as a forecast predictor unless recalculated per training fold.
    """
    if not file_path: return None, 'No validated historical pattern label file supplied'
    raw = load_csv(file_path)
    if id_col not in raw or label_col not in raw:
        raise ValueError(f'Pattern file must contain {id_col!r} and {label_col!r}')
    pat = raw[[id_col,label_col]].rename(columns={id_col:'adm_id',label_col:'pattern'}).copy()
    pat['adm_id'] = pat.adm_id.astype('string').str.strip()
    if pat.adm_id.isna().any() or pat.adm_id.duplicated().any():
        raise ValueError('Pattern labels must have exactly one record per administrative ID')
    joined = df.merge(pat,on='adm_id',how='left',validate='many_to_one')
    if joined.pattern.notna().sum() == 0:
        raise ValueError('No administrative codes match between pattern file and panel')
    rows=[]
    for (pattern), block in joined.dropna(subset=['pattern']).groupby('pattern'):
        row={'level':level,'pattern':str(pattern),'regions':int(block.adm_id.nunique()),'region_year_rows':len(block)}
        for c in ['flood__flood_days','rainfall','runoff','et0']:
            row[c+'_mean']=block[c].mean() if c in block else np.nan
        rows.append(row)
    return pd.DataFrame(rows),f'Pattern matched {joined.pattern.notna().sum()}/{len(joined)} region-year rows; full-period patterns are descriptive only'


def registry(df,level):
    # M3/M4: water level/discharge are NOT region-year validated in supplied panels.
    names = ['historical_flood_days','rainfall','runoff','et0','lake_water_level','dfo_discharge','historical_pattern']
    features = {'historical_flood_days':'flood__flood_days', 'rainfall':'rainfall','runoff':'runoff','et0':'et0'}
    rows=[]
    for n in names:
        col=features.get(n)
        valid=int(df[col].notna().sum()) if col and col in df else 0
        rows.append({'level':level,'feature':n,'source_column':col or '', 'valid_region_years':valid,
                     'total_region_years':len(df),'available_for_exploration':valid>0,
                     'forecast_policy':('LAG_ONE_YEAR_REQUIRED' if n in features else 'NOT_VALIDATED_FOR_FORECAST'),
                     'note':('Must reconstruct within each rolling training window' if n=='historical_pattern' else
                             'Needs audited admin/year and hydrologic linkage' if n in ('lake_water_level','dfo_discharge') else
                             '2025 ADM3 hydrology partial coverage must be retained' if n in ('rainfall','runoff') and level=='ADM3' else '')})
    return pd.DataFrame(rows)


def feature_sets(level, reg):
    present = reg.set_index('feature').valid_region_years.to_dict()
    sets = [
        ('M0',['historical_flood_days'],'Persistence-only benchmark'),
        ('M1',['historical_flood_days','rainfall','runoff'],'Adds rainfall + runoff'),
        ('M2',['historical_flood_days','rainfall','runoff','et0'],'Adds ET0'),
        ('M3',['historical_flood_days','rainfall','runoff','et0','lake_water_level'],'Requires spatially linked lake observations'),
        ('M4',['historical_flood_days','rainfall','runoff','et0','lake_water_level','dfo_discharge'],'Requires validated DFO discharge'),
        ('M5',['historical_flood_days','rainfall','runoff','et0','lake_water_level','dfo_discharge','historical_pattern'],'Requires fold-specific historical patterns')]
    return pd.DataFrame([{'level':level,'set':name,'features':'+'.join(cols),'purpose':why,
         'availability':'CANDIDATE_CHECK_MISSINGNESS' if all(present.get(c,0)>0 for c in cols) else 'BLOCKED_SOURCE_NOT_READY',
         'unavailable_features':'+'.join(c for c in cols if present.get(c,0)==0)} for name,cols,why in sets])


def plot_correlations(table, path, level):
    chosen = table[(table.relationship=='previous_year_to_current') & (table.outcome=='flood__flood_days')]
    fig, ax=plt.subplots(figsize=(7,4.5))
    if not chosen.empty and chosen.pearson_r.notna().any():
        ax.bar(chosen.feature, chosen.pearson_r)
        ax.axhline(0, color='black',lw=0.8)
        ax.set_ylim(-1,1)
        ax.set_ylabel('Pearson r (prior year to flood days)')
    else:
        ax.text(0.5,0.5,'No sufficient valid pairs',ha='center',va='center',transform=ax.transAxes)
        ax.set_xticks([])
    ax.set_title(f'{level} | Exploratory lagged driver associations')
    fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--adm2-pattern-file',type=Path,default=None)
    p.add_argument('--adm3-pattern-file',type=Path,default=None)
    p.add_argument('--pattern-id-col',default='adm_id')
    p.add_argument('--pattern-label-col',default='pattern')
    a=p.parse_args()
    root=a.root.expanduser().resolve()
    out=root/BASE/'03_flood_pattern_drivers'
    tables=out/'tables';figures=out/'figures'
    tables.mkdir(parents=True,exist_ok=True); figures.mkdir(parents=True,exist_ok=True)
    all_coverage=[];all_cor=[];all_reg=[];all_sets=[];notes=[]
    for level in ('ADM2','ADM3'):
        df=load_level(root,level)
        df.to_csv(tables/f'{level.lower()}_exploratory_panel.csv',index=False)
        cov=coverage(df,level);cors=correlations(df,level);reg=registry(df,level);sets=feature_sets(level,reg)
        all_coverage.append(cov);all_cor.append(cors);all_reg.append(reg);all_sets.append(sets)
        cors.to_csv(tables/f'driver_correlations_{level.lower()}.csv',index=False)
        plot_correlations(cors,figures/f'{level.lower()}_lagged_driver_correlations.png',level)
        pf=a.adm2_pattern_file if level=='ADM2' else a.adm3_pattern_file
        pat,note=external_patterns(df,level,pf,a.pattern_id_col,a.pattern_label_col)
        notes.append(f'{level}: {note}')
        if pat is not None: pat.to_csv(tables/f'pattern_driver_comparison_{level.lower()}.csv',index=False)
        print(f'{level}: {len(df)} region-year rows, {df.adm_id.nunique()} regions; '+
              ', '.join(f'{x}={df[x].notna().sum()}' for x in ['rainfall','runoff','et0']),flush=True)
    pd.concat(all_coverage).to_csv(tables/'data_coverage_summary.csv',index=False)
    allcorr=pd.concat(all_cor,ignore_index=True)
    allcorr.to_csv(tables/'all_driver_correlations.csv',index=False)
    pd.concat(all_reg).to_csv(tables/'feature_registry.csv',index=False)
    pd.concat(all_sets).to_csv(tables/'feature_sets_M0_M5.csv',index=False)
    comparison=allcorr[(allcorr.relationship=='previous_year_to_current') & (allcorr.outcome=='flood__flood_days')]
    comparison.to_csv(tables/'adm2_adm3_lagged_comparison.csv',index=False)
    notes.extend(['Same-year correlations describe associations only; not next-year forecasts.',
     'Lagged correlation uses t-1 drivers for flood outcome in t, with adjacent calendar years enforced.',
     'Within-admin demeaning is exploratory and does not control all time-varying confounders.',
     'M0-M5 definitions are proposed feature configurations, not fitted model scores.',
     'M3-M5 blocked until spatial and temporal data provenance independently validated.',
     'Precomputed full-period clusters/patterns MUST NOT be used as future-prediction features.',
     'Partial 2025 ADM3 hydrology remains observed-day totals, not complete annual totals.',
     'Do not interpret correlation or group differences as causal relationships.'])
    (out/'methodology_report.md').write_text('# JBG060 Script 03 — Pattern / Driver Exploration\n\n'+
        '\n'.join('- '+x for x in notes)+'\n',encoding='utf-8')
    (out/'run_summary.json').write_text(json.dumps({'script':'03','levels':['ADM2','ADM3'],
      'output':str(out),'limitations':notes},indent=2),encoding='utf-8')
    print('Output:',out,flush=True)

if __name__=='__main__': main()
