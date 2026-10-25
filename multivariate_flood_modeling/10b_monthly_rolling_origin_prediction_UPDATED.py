#!/usr/bin/env python3
"""JBG060 Script 10b — monthly rolling-origin vs fixed forecasting.

Run from VS Code without arguments:
  python scripts/multivariate_flood_modeling/10b_monthly_rolling_origin_prediction.py

Uses existing Script 07 (same folder), Script 05a monthly panels and the same
RandomForestRegressor setup as Script 07 (minimum leaf size=3).

Time design: fit 2002–2019; validation 2020–2022 (freeze warning thresholds);
test 2023–2025. Fixed is never refitted. Monthly Rolling is refitted each
month on all completely observed preceding months, starting 2002.

Month-memory designs:
 A = same calendar month in previous 1 and 2 years (lags 12, 24)
 B = previous 1, 2 and 3 calendar months (lags 1, 2, 3)
 C = A + B (lags 1, 2, 3, 12, 24)
Drivers M0–M4 follow Script 07. All 15 configurations use identical complete
case region-month observations. Any feature for forecast month t originates
strictly before month t. High-activity warning cutoffs are selected only on
validation, using fixed historical fits, then frozen for both test strategies.

The monthly flood_days target is an EXPLORATORY recorded flood-event-date
proxy, NOT validated inundation days or true disaster warning labels.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

COLORS = {'Fixed': '#7EC9F2', 'Monthly Rolling': '#7546B7'}
MEMORIES = {
    'A_same_month_previous_years': (12, 24),
    'B_recent_months': (1, 2, 3),
    'C_combined': (1, 2, 3, 12, 24),
}
MODEL_LABELS = {
    'M0_history': 'M0: Historical flood only',
    'M1_rainfall': 'M1: Flood + Rainfall',
    'M2_runoff': 'M2: Flood + Runoff',
    'M3_et0': 'M3: Flood + ET0',
    'M4_all_drivers': 'M4: Flood + Rainfall + Runoff + ET0',
}
MEMORY_LABELS = {
    'A_same_month_previous_years': 'A: Previous 1 & 2 years, same month',
    'B_recent_months': 'B: Previous 1, 2 & 3 months',
    'C_combined': 'C: Combined seasonal & recent memory',
}


def root_from_script():
    path = Path(__file__).resolve()
    if path.parent.name == 'multivariate_flood_modeling' and path.parent.parent.name == 'scripts':
        return path.parents[2]
    return Path.cwd()


def import_script07(path):
    if not path.is_file():
        raise FileNotFoundError(f'Script 07 not found: {path}. Place 10b beside 07_final_multidriver_flood_prediction.py')
    spec = importlib.util.spec_from_file_location('jbg_script07_for_10b', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_calendar_lags(df, variables):
    data = df.sort_values(['adm_id', 'year', 'month']).copy().reset_index(drop=True)
    data['period'] = data.year * 12 + data.month
    groups = data.groupby('adm_id', sort=False)
    for lag in (1, 2, 3, 12, 24):
        actual_period = groups['period'].shift(lag)
        contiguous = actual_period.eq(data.period - lag)
        for var in variables:
            data[f'{var}_lag{lag}'] = groups[var].shift(lag).where(contiguous)
    return data


def cols_for(s07, memory, model):
    lags = MEMORIES[memory]
    drivers = ('flood_days',) + tuple(s07.MODELS[model])
    return ['month'] + [f'{field}_lag{lag}' for field in drivers for lag in lags]


def fit(data, columns, trees, seed):
    reg = RandomForestRegressor(n_estimators=trees, min_samples_leaf=3,
                                random_state=seed, n_jobs=-1)
    reg.fit(data[columns], data['flood_days'].to_numpy(float))
    return reg


def predict(model, data, columns):
    days = pd.to_datetime(dict(year=data.year, month=data.month, day=1)).dt.days_in_month.to_numpy()
    return np.clip(model.predict(data[columns]), 0, days)


def calculate(s07, y, yhat, event_cut, warning_cut):
    return {**s07.regression_metrics(y, yhat),
            **s07.classify_metrics(y, yhat, event_cut, warning_cut)}


def safe_rows(s07, panel, level, trees, seed, quantile, min_precision, selected):
    raw, info = s07.load_panel(panel, level)
    source = make_calendar_lags(raw, ('flood_days',) + tuple(s07.DRIVERS))
    # Comparisons use the SAME complete-case cohort for A/B/C and M0-M4.
    needed = ['flood_days'] + cols_for(s07, 'C_combined', 'M4_all_drivers')
    data = source.dropna(subset=needed).copy()
    data = data[data.year.between(2002, 2025)].sort_values(['year','month','adm_id']).reset_index(drop=True)
    train = data[data.year.between(2002, 2019)].copy()
    val = data[data.year.between(2020, 2022)].copy()
    test = data[data.year.between(2023, 2025)].copy()
    if min(len(train), len(val), len(test)) < 10:
        raise ValueError(f'{level}: insufficient rows train={len(train)}, val={len(val)}, test={len(test)}')
    pos = train.loc[train.flood_days.gt(0), 'flood_days']
    if pos.empty:
        raise ValueError(f'{level}: no positive training targets; cannot set observed event threshold')
    event_cut = float(math.ceil(max(1.0, pos.quantile(quantile))))
    combinations = [(memory, model) for memory in MEMORIES for model in selected]
    periods = sorted(test.period.unique())
    predictions, threshold_rows, score_rows = [], [], []
    fixed_models = {}
    thresholds = {}

    for memory, model in combinations:
        cols = cols_for(s07, memory, model)
        fixed = fit(train, cols, trees, seed)
        val_pred = predict(fixed, val, cols)
        selection = s07.pick_alert_threshold(val.flood_days.to_numpy(float), val_pred,
                                             event_cut, min_precision)
        threshold = float(selection['alert_threshold'])
        fixed_models[memory, model] = fixed
        thresholds[memory, model] = threshold
        threshold_rows.append(dict(level=level, memory=memory, model=model,
                                   event_threshold=event_cut, warning_threshold=threshold,
                                   validation_F2=selection.get('validation_F2'),
                                   selection_note=selection.get('selection_note'),
                                   training_years='2002-2019', validation_years='2020-2022'))
    print(f'{level}: trained {len(combinations)} fixed models, selected validation warning thresholds', flush=True)

    for count, period in enumerate(periods, start=1):
        target = test.loc[test.period.eq(period)].copy()
        history = data.loc[data.period.lt(period)].copy()
        if len(target) == 0 or len(history) == 0:
            continue
        for memory, model in combinations:
            cols = cols_for(s07, memory, model)
            # NOTE: Rolling is strictly recalibrated using data BEFORE origin.
            rolling = fit(history, cols, trees, seed)
            for strategy, reg in [('Fixed', fixed_models[memory, model]),
                                  ('Monthly Rolling', rolling)]:
                predicted = predict(reg, target, cols)
                obs = target.flood_days.to_numpy(float)
                warn_cut = thresholds[memory, model]
                true_high = (obs >= event_cut).astype(int)
                warning = (predicted >= warn_cut).astype(int)
                group = pd.DataFrame({
                    'level': level, 'memory': memory, 'model': model,
                    'strategy': strategy, 'adm_id': target.adm_id.to_numpy(),
                    'year': target.year.to_numpy(int), 'month': target.month.to_numpy(int),
                    'actual_event_dates': obs, 'predicted_event_dates': predicted,
                    'actual_high_activity': true_high, 'predicted_warning': warning,
                    'event_threshold': event_cut, 'warning_threshold': warn_cut,
                    'origin_training_last_period': 2019*12+12 if strategy == 'Fixed' else period-1,
                })
                predictions.append(group)
                score_rows.append(dict(level=level, memory=memory, model=model,
                                       strategy=strategy, year=int(target.year.iloc[0]),
                                       month=int(target.month.iloc[0]),
                                       train_n=len(train) if strategy == 'Fixed' else len(history),
                                       test_n=len(target),
                                       **calculate(s07, obs, predicted, event_cut, warn_cut)))
        if count % 6 == 0 or count == len(periods):
            print(f'{level}: completed {count}/{len(periods)} monthly test origins', flush=True)

    audit = dict(**info, comparable_rows=len(data), train_n=len(train),
                 validation_n=len(val), test_n=len(test), test_months=len(periods),
                 event_threshold=event_cut, models=selected, trees=trees)
    return (pd.concat(predictions, ignore_index=True), pd.DataFrame(score_rows),
            pd.DataFrame(threshold_rows), audit)


def pooled(s07, pred):
    results = []
    for (level, memory, model, strategy), part in pred.groupby(
        ['level', 'memory', 'model', 'strategy'], sort=False
    ):
        m = calculate(s07, part.actual_event_dates.to_numpy(float),
                      part.predicted_event_dates.to_numpy(float),
                      float(part.event_threshold.iloc[0]), float(part.warning_threshold.iloc[0]))
        results.append(dict(level=level, memory=memory, model=model, strategy=strategy,
                            test_n=len(part), **m))
    return pd.DataFrame(results)


def yearly(s07, pred):
    results = []
    for (level, memory, model, strategy, year), part in pred.groupby(
        ['level', 'memory', 'model', 'strategy', 'year'], sort=False
    ):
        m = calculate(s07, part.actual_event_dates.to_numpy(float),
                      part.predicted_event_dates.to_numpy(float),
                      float(part.event_threshold.iloc[0]), float(part.warning_threshold.iloc[0]))
        results.append(dict(level=level, memory=memory, model=model, strategy=strategy,
                            year=year, test_n=len(part), **m))
    return pd.DataFrame(results)


def explanatory_footer(fig):
    fig.text(.5, .035, 'M0 Flood history  |  M1 + Rainfall  |  M2 + Runoff  |  M3 + ET0  |  M4 + Rainfall + Runoff + ET0',
             ha='center', va='center', fontsize=10, weight='bold', color='#34303A')
    fig.text(.5, .014, 'Recorded flood-event-date PROXY (not validated inundation duration). 2023–2025 test; same comparable ADM-months.',
             ha='center', va='center', fontsize=8.5, color='#66616C')


def annotated_bars(ax, x, width, values, label, color, percent=False):
    bars = ax.bar(x, values, width, label=label, color=color, alpha=.96)
    for bar, value in zip(bars, values):
        shown = value*100 if percent else value
        ax.annotate(f'{shown:.1f}{"%" if percent else ""}',
                    (bar.get_x()+bar.get_width()/2, bar.get_height()), xytext=(0, 4),
                    textcoords='offset points', ha='center', va='bottom', fontsize=8,
                    rotation=0, clip_on=False)
    return bars


def plot_memory_comparison(results, level, memory, out):
    data = results[(results.level == level) & (results.memory == memory)]
    names = [m for m in MODEL_LABELS if m in set(data.model)]
    fig, axes = plt.subplots(1, 3, figsize=(17, 7.0))
    fig.subplots_adjust(left=.07, right=.985, top=.81, bottom=.21, wspace=.30)
    x = np.arange(len(names))
    for ax, (metric, percent) in zip(axes, [('MAE',False),('Recall',True),('F1',True)]):
        for i, strategy in enumerate(COLORS):
            d = data[data.strategy.eq(strategy)].set_index('model')
            values = [float(d.loc[n,metric]) for n in names]
            annotated_bars(ax, x+(i-.5)*.36, .34, values, strategy, COLORS[strategy], percent)
        ax.set_title(f'{metric}{" (%)" if percent else " (event dates)"}', fontsize=12, weight='bold')
        ax.set_xticks(x, [m[:2] for m in names])
        ax.grid(axis='y', alpha=.16)
        ax.set_axisbelow(True)
        vmax = float(data[metric].max())
        ax.set_ylim(0, (max(1.05, vmax*1.2) if percent else max(.25, vmax*1.2)))
    axes[0].legend(frameon=False, loc='upper left')
    fig.suptitle(f'{level} | Fixed vs Monthly Rolling-Origin\n{MEMORY_LABELS[memory]}',
                 fontsize=17, weight='bold', y=.965)
    explanatory_footer(fig)
    stem = f'{level.lower()}_{memory.split("_")[0]}_model_comparison'
    for ext in ('png','pdf'):
        fig.savefig(out/f'{stem}.{ext}', dpi=230, facecolor='white')
    plt.close(fig)


def plot_m4_memory(results, level, out):
    data = results[(results.level == level) & (results.model == 'M4_all_drivers')]
    if data.empty: return
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.1))
    fig.subplots_adjust(left=.08, right=.985, top=.82, bottom=.25, wspace=.3)
    ms = list(MEMORIES)
    for ax, (metric, percent) in zip(axes, [('MAE',False),('Recall',True),('F1',True)]):
        for i, strategy in enumerate(COLORS):
            d = data[data.strategy.eq(strategy)].set_index('memory')
            values = [float(d.loc[m, metric]) for m in ms]
            annotated_bars(ax, np.arange(3)+(i-.5)*.36, .34, values,
                           strategy, COLORS[strategy], percent)
        ax.set_xticks(np.arange(3), ['A','B','C'])
        ax.set_title(f'{metric}{" (%)" if percent else " (event dates)"}', weight='bold')
        ax.grid(axis='y', alpha=.16); ax.set_axisbelow(True)
        vmax = float(data[metric].max())
        ax.set_ylim(0, (max(1.05, vmax*1.2) if percent else max(.25, vmax*1.2)))
    axes[0].legend(frameon=False)
    fig.suptitle(f'{level} | M4: Flood + Rainfall + Runoff + ET0\nWhich month-memory design works best?',
                 weight='bold', fontsize=16, y=.96)
    fig.text(.5,.11,'A  Previous years, same month    |    B  Previous 1–3 months    |    C  Combined A + B',
             ha='center',fontsize=10,weight='bold')
    fig.text(.5,.057,'Every bar is labelled with its actual test metric; 2023–2025.',ha='center', fontsize=9)
    for ext in ('png','pdf'):
        fig.savefig(out/f'{level.lower()}_M4_month_memory_comparison.{ext}',dpi=230,facecolor='white')
    plt.close(fig)


def plot_monthly_mae_36months(monthly, level, out):
    sub = monthly[(monthly.level == level) & (monthly.model == 'M4_all_drivers') & (monthly.memory == 'C_combined')].copy()
    if sub.empty:
        return
    sub['date'] = pd.to_datetime(dict(year=sub.year.astype(int), month=sub.month.astype(int), day=1))
    ordered = sorted(sub['date'].unique())
    labels = [pd.Timestamp(d).strftime('%Y-%m') for d in ordered]
    x = np.arange(len(ordered))

    fig, ax = plt.subplots(figsize=(23, 7.5))
    fig.subplots_adjust(left=.055, right=.995, top=.84, bottom=.24)
    for strategy, color in COLORS.items():
        series = sub[sub.strategy.eq(strategy)].sort_values('date')
        y = series['MAE'].to_numpy(dtype=float)
        ax.plot(x, y, 'o-', lw=2.2, ms=5.8, color=color, label=strategy)
        for xx, yy in zip(x, y):
            offset = 8 if strategy == 'Fixed' else -14
            ax.annotate(f'{yy:.2f}', (xx, yy), xytext=(0, offset), textcoords='offset points',
                        ha='center', fontsize=7.5, color=color, weight='bold', clip_on=False)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel('MAE (event-date proxy days)', weight='bold')
    ax.set_xlabel('Test month', weight='bold')
    ax.grid(axis='y', alpha=.20)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc='upper right')
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(max(0, ymin - (ymax-ymin)*0.04), ymax + (ymax-ymin)*0.16)
    fig.suptitle(f'{level} | M4 + Combined Month Memory (C)\nMonthly Prediction Performance (2023–2025)',
                 fontsize=17, weight='bold', y=.965)
    fig.text(.5, .085, 'Monthly test metric from 36 prediction months. M4 = Flood history + Rainfall + Runoff + ET0.',
             ha='center', fontsize=10, weight='bold')
    fig.text(.5, .058, 'Fixed = one model trained on 2002–2019. Monthly Rolling = refit every month using all months available before the prediction month.',
             ha='center', fontsize=9)
    fig.text(.5, .033, 'Target = recorded flood-event-date proxy only; not validated inundation duration or operational warning quality.',
             ha='center', fontsize=8.7, color='#66616C')
    for ext in ('png','pdf'):
        fig.savefig(out / f'{level.lower()}_M4_C_monthly_MAE_36months.{ext}', dpi=230, facecolor='white')
    plt.close(fig)


def plot_annual_summary(y, level, out):
    sub = y[(y.level == level) & (y.model == 'M4_all_drivers') & (y.memory == 'C_combined')].copy()
    if sub.empty:
        return
    years = sorted(sub.year.astype(int).unique())
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 10.2))
    fig.subplots_adjust(left=.07, right=.985, top=.85, bottom=.16, hspace=.34, wspace=.24)
    metrics = [('Recall', True), ('F1', True), ('FP', False), ('FN', False)]
    x = np.arange(len(years))
    width = 0.36
    axes = axes.flatten()
    for ax, (metric, percent) in zip(axes, metrics):
        for i, (strategy, color) in enumerate(COLORS.items()):
            d = sub[sub.strategy.eq(strategy)].set_index('year').reindex(years)
            values = d[metric].astype(float).to_numpy()
            bars = ax.bar(x + (i - 0.5) * width, values, width=width*0.92, color=color, label=strategy, alpha=.96)
            for bar, value in zip(bars, values):
                label = f'{value*100:.1f}%' if percent else f'{int(value)}'
                ax.annotate(label, (bar.get_x() + bar.get_width()/2, bar.get_height()),
                            xytext=(0, 4), textcoords='offset points', ha='center', va='bottom',
                            fontsize=9, weight='bold', color=color)
        ax.set_xticks(x, [str(y0) for y0 in years])
        ax.set_title(metric + (' (%)' if percent else ' (count)'), weight='bold')
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
        vmax = float(sub[metric].max())
        ax.set_ylim(0, max(1.05, vmax*1.23) if percent else max(1.0, vmax*1.22))
    axes[0].legend(frameon=False, ncol=2, loc='upper left')
    fig.suptitle(f'{level} | M4 + Combined Month Memory (C)\nAnnual Performance Summary (2023–2025)',
                 fontsize=17, weight='bold', y=.965)
    fig.text(.5, .085, 'Yearly summary from monthly predictions. Bars show Recall, F1, False Positives and False Negatives for Fixed vs Monthly Rolling.',
             ha='center', fontsize=10, weight='bold')
    fig.text(.5, .058, 'M4 = Flood history + Rainfall + Runoff + ET0. C = previous 1–3 months + previous 1 & 2 years, same month.',
             ha='center', fontsize=9.4)
    fig.text(.5, .033, 'Target = recorded flood-event-date proxy only; not validated inundation duration or operational warning quality.',
             ha='center', fontsize=8.7, color='#66616C')
    for ext in ('png','pdf'):
        fig.savefig(out / f'{level.lower()}_M4_C_annual_summary.{ext}', dpi=230, facecolor='white')
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    root=root_from_script()
    ap.add_argument('--root',type=Path,default=root)
    ap.add_argument('--level',choices=['ADM2','ADM3','both'],default='ADM3',
                    help='ADM3 by default; choose both to run ADM2 too')
    ap.add_argument('--trees',type=int,default=50,
                    help='RF trees per fit. Monthly refitting is computationally expensive.')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--event-quantile',type=float,default=.75)
    ap.add_argument('--min-precision',type=float,default=.30)
    ap.add_argument('--models',default='M0,M1,M2,M3,M4',
                    help='Optional speed test e.g. --models M0,M4')
    ap.add_argument('--output',type=Path,default=None)
    args=ap.parse_args()
    if args.trees < 1: ap.error('--trees must be >= 1')
    if not 0<args.event_quantile<1 or not 0<=args.min_precision<=1:
        ap.error('invalid event-quantile or min-precision')
    root=args.root.resolve()
    out=(args.output or root/'processed_data/multivariate_flood_modeling/10b_monthly_rolling_origin').resolve()
    out.mkdir(parents=True,exist_ok=True)
    s07=import_script07(root/'scripts/multivariate_flood_modeling/07_final_multidriver_flood_prediction.py')
    selected=[]
    for tag in args.models.split(','):
        tag=tag.strip().upper()
        found=[m for m in s07.MODELS if m.startswith(tag+'_')]
        if len(found)!=1: ap.error(f'Unknown model {tag}; choose M0..M4')
        selected.append(found[0])
    selected=list(dict.fromkeys(selected))
    levels=['ADM2','ADM3'] if args.level=='both' else [args.level]
    predictions,monthly_scores,thresholds,audits=[],[],[],[]
    for level in levels:
        path=root/f'processed_data/multivariate_flood_modeling/05a_monthly_panel/{level.lower()}_monthly_panel.csv'
        result=safe_rows(s07,path,level,args.trees,args.seed,args.event_quantile,args.min_precision,selected)
        p,m,t,a=result
        predictions.append(p); monthly_scores.append(m); thresholds.append(t); audits.append(a)
        print(f'{level}: finished; saving results...',flush=True)
    pred=pd.concat(predictions,ignore_index=True)
    by_month=pd.concat(monthly_scores,ignore_index=True)
    th=pd.concat(thresholds,ignore_index=True)
    aggregate=pooled(s07,pred)
    by_year=yearly(s07,pred)
    pred.to_csv(out/'01_test_predictions_monthly.csv',index=False)
    by_month.to_csv(out/'02_metrics_by_test_month.csv',index=False)
    by_year.to_csv(out/'03_metrics_by_test_year.csv',index=False)
    aggregate.to_csv(out/'04_pooled_test_metrics.csv',index=False)
    th.to_csv(out/'05_validation_thresholds.csv',index=False)
    pd.DataFrame(audits).to_csv(out/'06_cohort_audit.csv',index=False)
    metadata={
        'target':'EXPLORATORY_EVENT_DATE_PROXY: regional recorded flood-event dates, not true inundation duration',
        'train':'2002-2019 fixed fit (lag24 means first evaluable year is 2002)',
        'validation':'2020-2022; warning thresholds chosen using fixed-trained models; frozen for both strategies',
        'test':'2023-2025; monthly prediction origins; test never tunes thresholds',
        'fixed':'fitted once on 2002-2019 and never refitted',
        'monthly_rolling':'at month t refit with all labels through t-1 only; no target-month meteorology',
        'month_memory_designs':{k:list(v) for k,v in MEMORIES.items()},
        'driver_models':{k:list(v) for k,v in s07.MODELS.items()},
        'model_explanations':MODEL_LABELS,
        'common_cohort':'All combinations use complete cases for lags 1,2,3,12,24 of history, rainfall, runoff and ET0',
        'trees':args.trees,'seed':args.seed,'event_quantile':args.event_quantile,
        'minimum_validation_precision':args.min_precision,
        'important_limitations':[
            'Rolling is an updating strategy, not an independent new holdout; 2023-2025 previously reviewed during project development',
            'Future monthly rainfall/runoff/ET0 actual observations are never included as same-month features',
            'Data availability/processing delays at deployment are not modelled',
            'Lake water levels and DFO are not included (hydrologic linkage still unverified)',
            'Comparisons are within administrative level, not a spatial out-of-region holdout',
            'The flood target is a proxy and not validated humanitarian warning performance',
        ]}
    (out/'07_methodology.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    for level in levels:
        for memory in MEMORIES:
            plot_memory_comparison(aggregate,level,memory,out)
        plot_m4_memory(aggregate,level,out)
        plot_monthly_mae_36months(by_month, level, out)
        plot_annual_summary(by_year, level, out)
    print('\nSaved:',out)
    print(aggregate[['level','memory','model','strategy','MAE','Recall','F1','FP','FN']].to_string(index=False))
    print('\nINTERPRETATION: event-date proxy only; not verified flood duration or operational warning quality.')

if __name__=='__main__':
    main()
