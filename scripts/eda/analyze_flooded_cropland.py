import sys
import os
# Ensure script runs from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import rasterio
import os
import matplotlib.pyplot as plt
import seaborn as sns

from processing_data.loading import load_flood_masks

os.makedirs('outputs/eda_graphs', exist_ok=True)

# peak flood years vs baseline
years = range(2018, 2025)

# south sudan bbox roughly
bbox_ssd = {
    'lon_min': 24, 'lat_min': 3,
    'lon_max': 36, 'lat_max': 13
}

crop_raster_path = './raw_data/farmland/asap_mask_crop_v04.tif'
results = []

with rasterio.open(crop_raster_path) as src:
    for y in years:
        print(f"doing {y}...")
        df_floods = load_flood_masks([y], bbox=bbox_ssd)
        
        if df_floods.empty:
            results.append({'Year': y, 'Flooded_Cropland_Score': 0})
            continue
            
        # get unique flood pixels
        unique_floods = df_floods[['lat', 'lon']].drop_duplicates()
        coords = [(row.lon, row.lat) for _, row in unique_floods.iterrows()]
        
        flood_crop_score = 0
        
        for val in src.sample(coords):
            v = val[0]
            # check if valid percentage
            if src.nodata is None or v != src.nodata:
                if not np.isnan(v) and v > 0:
                    flood_crop_score += (v / 100.0) 
                    
        results.append({'Year': y, 'Flooded_Cropland_Score': flood_crop_score})

df_res = pd.DataFrame(results)
df_res.to_csv('outputs/eda_graphs/flooded_cropland_extent_2018_2024.csv', index=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=df_res, x='Year', y='Flooded_Cropland_Score', color='#d9534f')
plt.title('Flooded Cropland Extent (2018–2024)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Submerged Ag Area', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, row in df_res.iterrows():
    plt.text(i, row['Flooded_Cropland_Score'] + (df_res['Flooded_Cropland_Score'].max() * 0.02), 
             f"{int(row['Flooded_Cropland_Score'])}", color='black', ha="center")
             
plt.tight_layout()
plt.savefig('outputs/eda_graphs/Flooded_Cropland_Extent_2018_2024.png', dpi=300)
plt.close()
print("done")


