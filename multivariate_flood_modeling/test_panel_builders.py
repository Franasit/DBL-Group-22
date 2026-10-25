"""Synthetic smoke tests for JBG060 panel builders; does not certify real source values."""
from pathlib import Path
import tempfile
import pandas as pd
from panel_builder_common import build, TARGET_PATHS


def test_level(root: Path, level: str, n: int):
    ids = [f'SS{i:04d}' for i in range(n)]
    rows = []
    for uid in ids:
        for year in range(2000, 2026):
            rows.append({f'{level}_ID': uid, 'year': year, 'flood_days': 0 if year == 2000 else 12, 'annual_flooded_ratio': .1})
    target = root / TARGET_PATHS[level]
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(target, index=False)
    if level == 'ADM2':
        hyd = root / 'processed_data/cross_dataset_analysis/adm2_hydrology/ADM2_annual_rainfall_runoff.csv'
        hyd.parent.mkdir(parents=True, exist_ok=True)
        # Uses alternative source IDs; shared standardized ADM2_PCODE is the bridge.
        pd.DataFrame([{'ADM2_ID': str(i+1), 'ADM2_PCODE':ids[i], 'year':year,
                       'annual_rainfall_mm': 300+year%20, 'annual_runoff_mm':80}
                      for i in range(n) for year in range(2000,2026)]).to_csv(hyd, index=False)
        # Add target-side standardized code to test a source->target crosswalk.
        df = pd.read_csv(target)
        df['ADM2_PCODE'] = df['ADM2_ID']
        df.to_csv(target,index=False)
    result = build(root, level, n)
    assert result['panel_rows'] == n * 26
    d = root / f'processed_data/multivariate_flood_modeling/{"01_adm2" if level=="ADM2" else "02_adm3"}_panel_dataset'
    pred = pd.read_csv(d/'prediction_ready_panel.csv')
    assert pred.loc[pred.year.eq(2000), 'flood__flood_days__lag1'].isna().all()
    assert (pred.loc[pred.year.eq(2001), 'flood__flood_days__lag1'] == 0).all()
    if level == 'ADM2':
        joins = pd.read_csv(d/'join_quality_report.csv')
        assert joins.matched_keys.iloc[0] == n * 26
    print('PASS', level, n * 26)


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        test_level(root,'ADM2',79)
        test_level(root,'ADM3',512)
