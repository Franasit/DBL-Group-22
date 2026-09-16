"""
Data Cleaning Script

This script cleans and merges data from the following folders:
- Input: './raw_data/flood_masks/' (via load_flood_masks)
- Input: './raw_data/worldpop/'
- Output: './processing_data/cleaned/'
"""
import sys
import os
# Ensure script runs from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import rasterio
import os
from pathlib import Path
from processing_data.loading import load_flood_masks

# Sudd wetlands area
bbox = {
    'lat_min': 8.5, 'lat_max': 10.0,
    'lon_min': 29.5, 'lon_max': 32.5
}
target_yr = 2020

out_dir = Path('./processing_data/cleaned')
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f'cleaned_flood_exposure_{target_yr}.csv'

print(f"--- Data Cleaning Process ---")
print(f"Input flood masks folder: ./raw_data/flood_masks/")
print(f"Input population folder: ./raw_data/worldpop/")
print(f"Output folder for cleaned data: {out_dir}")
print(f"-----------------------------")

print(f"loading masks for {target_yr}")
df_floods = load_flood_masks([target_yr], bbox=bbox)

if df_floods.empty:
    print("no data")
    exit()

pop_raster = f'./raw_data/worldpop/ssd_pop_{target_yr}_CN_100m_R2025A_v1.tif'

unique_locs = df_floods[['lat', 'lon']].drop_duplicates().copy()
unique_locs['population'] = np.nan

print("sampling population...")
with rasterio.open(pop_raster) as src:
    coords = [(lon, lat) for lon, lat in zip(unique_locs['lon'], unique_locs['lat'])]
    
    pop_vals = []
    for val in src.sample(coords):
        v = val[0]
        # handle missing data
        if src.nodata is not None and v == src.nodata:
            pop_vals.append(np.nan)
        elif v < 0:
            pop_vals.append(np.nan)
        else:
            pop_vals.append(v)
            
    unique_locs['population'] = pop_vals

df_clean = pd.merge(df_floods, unique_locs, on=['lat', 'lon'], how='left')

# drop unpopulated/nodata pixels
df_clean = df_clean.dropna(subset=['population'])
df_clean.to_csv(out_path, index=False)
print("saved to csv")

