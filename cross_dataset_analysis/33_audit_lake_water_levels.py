"""Script 33 — Satellite lake water-level audit for Victoria, Kyoga, Albert.
Uses observed altimetry heights (metres), NOT river discharge or spatially assigned
ADM3 predictors. Returns satellite observations, monthly/annual summaries and QC.
Requires: pandas numpy matplotlib h5py (for netCDF4/HDF5 Albert file).
"""
from pathlib import Path
import io, zipfile
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path('/Users/shibai/Documents/GitHub/DBL-Group-22')
INPUT=ROOT/'raw_data'/'water_levels_lakes'
OUTPUT=ROOT/'processed_data'/'cross_dataset_analysis'/'05_hydrological_drivers'/'33_lake_water_levels'
TABLE=OUTPUT/'tables';FIG=OUTPUT/'figures'
for p in (TABLE,FIG):p.mkdir(parents=True,exist_ok=True)


def files():
    if INPUT.is_dir():
        for p in INPUT.rglob('*'):
            if p.suffix.lower() in ('.txt','.nc'):yield p.name,p.read_bytes()
    elif INPUT.with_suffix('.zip').exists():
        with zipfile.ZipFile(INPUT.with_suffix('.zip')) as z:
            for n in z.namelist():
                if not n.startswith('__MACOSX/') and Path(n).suffix.lower() in ('.txt','.nc'):
                    yield Path(n).name,z.read(n)
    else:raise FileNotFoundError(f'Unzip to {INPUT}, or put {INPUT.with_suffix(".zip")}')


def parse_text(data,lake):
    out=[]
    for line in data.decode('utf-8',errors='replace').splitlines():
        a=line.split()
        if len(a)!=16 or not a[2].isdigit() or len(a[2])!=8:continue
        date=pd.to_datetime(a[2],format='%Y%m%d',errors='coerce')
        try: height=float(a[14]);relative=float(a[5]);error=float(a[6]);flag=int(a[15])
        except ValueError:continue
        # Column 15 is EGM2008 orthometric height, 9999.99 means missing.
        if height>=9999 or relative>=999 or error>=99:height=np.nan
        out.append({'lake':lake,'date':date,'water_level_m':height,'error_m':error if error<99 else np.nan,
                    'satellite':a[0],'source_flag':flag,'datum':'EGM2008','source_type':'altimetry_txt'})
    return pd.DataFrame(out)


def parse_albert(data):
    with h5py.File(io.BytesIO(data),'r') as f:
        need=['datetime','water_level','error']
        if any(n not in f for n in need):raise ValueError(f'Albert NetCDF variables missing: {list(f.keys())}')
        dates=[v.decode() if isinstance(v,bytes) else str(v) for v in f['datetime'][:]]
        level=np.asarray(f['water_level'][:],dtype=float)
        errors=np.asarray(f['error'][:],dtype=float)
        attrs=dict(f['water_level'].attrs)
    valid_min=float(np.asarray(attrs.get('valid_min',[-np.inf])).ravel()[0])
    valid_max=float(np.asarray(attrs.get('valid_max',[np.inf])).ravel()[0])
    bad=(~np.isfinite(level)) | (level<valid_min) | (level>valid_max)
    level[bad]=np.nan
    return pd.DataFrame({'lake':'Albert','date':pd.to_datetime(dates,errors='coerce'),
       'water_level_m':level,'error_m':errors,'satellite':'unspecified','source_flag':np.nan,
       'datum':'NetCDF source datum (verify metadata)','source_type':'altimetry_netcdf'})


def main():
    datasets=[]
    for name,bytes_ in files():
        lower=name.lower()
        if 'victoria' in lower and name.endswith('.txt'):datasets.append(parse_text(bytes_,'Victoria'))
        elif 'kyoga' in lower and name.endswith('.txt'):datasets.append(parse_text(bytes_,'Kyoga'))
        elif 'albert' in lower and name.endswith('.nc'):datasets.append(parse_albert(bytes_))
    if len(datasets)!=3:raise ValueError(f'Expected 3 lake sources, got {len(datasets)}; verify filenames')
    df=pd.concat(datasets,ignore_index=True)
    df=df[df.date.between('2000-01-01','2025-12-31')].copy()
    df=df.sort_values(['lake','date'])
    df['year']=df.date.dt.year;df['month']=df.date.dt.month
    df['period']=df.date.dt.to_period('M')
    # Within-lake anomalies: no cross-lake comparison of absolute datum heights.
    monthly=df.groupby(['lake','period'],as_index=False).agg(
      water_level_m=('water_level_m','median'),observations=('water_level_m','count'),
      median_error_m=('error_m','median'))
    monthly['year']=monthly.period.dt.year;monthly['month']=monthly.period.dt.month
    baseline=monthly[monthly.year.between(2000,2019)].groupby(['lake','month']).water_level_m.agg(['mean','std','count']).reset_index()
    baseline.columns=['lake','month','baseline_mean_m','baseline_sd_m','baseline_years']
    monthly=monthly.merge(baseline,on=['lake','month'],how='left',validate='many_to_one')
    monthly['lake_level_anomaly_m']=monthly.water_level_m-monthly.baseline_mean_m
    monthly['lake_level_z']=np.where((monthly.baseline_sd_m>0)&(monthly.baseline_years>=10),
                                    monthly.lake_level_anomaly_m/monthly.baseline_sd_m,np.nan)
    monthly['anomaly_qc_flag']=np.select([monthly.baseline_years.fillna(0).lt(10),monthly.baseline_sd_m.fillna(0).le(0),monthly.lake_level_z.abs().gt(5)],['insufficient_baseline','zero_baseline_sd','extreme_z_review'],default='ok')
    # Preserve missing months instead of silently omitting them from the time series.
    calendar=pd.MultiIndex.from_product([sorted(df.lake.unique()),pd.period_range('2000-01','2025-12',freq='M')],names=['lake','period']).to_frame(index=False)
    monthly=calendar.merge(monthly,on=['lake','period'],how='left',validate='one_to_one')
    monthly['year']=monthly.period.dt.year;monthly['month']=monthly.period.dt.month
    monthly['observations']=monthly.observations.fillna(0).astype(int)
    monthly['anomaly_qc_flag']=np.where(monthly.observations.eq(0),'missing_month',monthly.anomaly_qc_flag)
    annual=df.groupby(['lake','year'],as_index=False).agg(
      median_height_m=('water_level_m','median'),valid_observations=('water_level_m','count'))
    qc=df.groupby('lake',as_index=False).agg(rows=('date','size'),valid_heights=('water_level_m','count'),
         missing_heights=('water_level_m',lambda s:int(s.isna().sum())),first_date=('date','min'),last_date=('date','max'))
    coverage=monthly.groupby('lake',as_index=False).agg(months_total=('period','size'),months_observed=('observations',lambda x:int(x.gt(0).sum())),extreme_z_months=('anomaly_qc_flag',lambda x:int(x.eq('extreme_z_review').sum())))
    qc=qc.merge(coverage,on='lake',validate='one_to_one')
    qc['monthly_coverage_fraction']=qc.months_observed/qc.months_total
    qc['baseline_note']='Seasonal z scores require >=10 valid monthly baseline years; zero months remain missing'
    qc['datum_warning']='Albert datum not assumed equal to EGM2008; compare anomalies, not absolute heights'
    df.drop(columns='period').to_csv(TABLE/'33_01_lake_observations_2000_2025.csv',index=False)
    monthly.assign(period=monthly.period.astype(str)).to_csv(TABLE/'33_02_lake_monthly_anomalies.csv',index=False)
    annual.to_csv(TABLE/'33_03_lake_annual_levels.csv',index=False)
    qc.to_csv(TABLE/'33_04_lake_qc.csv',index=False)
    monthly.groupby(['lake','anomaly_qc_flag'],dropna=False).size().rename('months').reset_index().to_csv(TABLE/'33_05_anomaly_qc_flags.csv',index=False)
    for lake,g in monthly.groupby('lake'):
        fig,ax=plt.subplots(figsize=(11,3.5))
        ax.plot(pd.to_datetime(g.period.astype(str)),g.lake_level_anomaly_m,marker='.',markersize=2,linewidth=.7)
        ax.axhline(0,color='grey',linewidth=.7);ax.set_title(f'{lake} — monthly lake-height anomaly (seasonal baseline 2000–2019)')
        ax.set_ylabel('Within-lake height anomaly (m)');fig.tight_layout()
        fig.savefig(FIG/f'33_{lake.lower()}_anomalies.png',dpi=170);plt.close(fig)
    print(f'Script 33 complete: {OUTPUT}. Review lake datum, temporal gaps, and provenance before causal claims.')
if __name__=='__main__':main()
