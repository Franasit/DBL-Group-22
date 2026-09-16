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
from processing_data.loading_impact_data import load_worldpop_area

os.makedirs('outputs/eda_graphs', exist_ok=True)
years = np.arange(2015, 2026)
    
bbox = {
    'lat_min': 8.5, 'lat_max': 10.0,
    'lon_min': 29.5, 'lon_max': 32.5
}
    
df_floods = load_flood_masks(years, bbox=bbox)
    
if not df_floods.empty:
    df_floods['year'] = df_floods['date'].dt.year
    
    # plot 1
    plt.figure(figsize=(10, 6))
    f_per_year = df_floods.groupby('year').size()
    sns.barplot(x=f_per_year.index, y=f_per_year.values, color="blue")
    plt.title('Total Flood Events per Year')
    plt.savefig('outputs/eda_graphs/flood_events_per_year.png')
    plt.close()
    
    # plot 2
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_floods, x='year', hue='flood_type')
    plt.title('Recurring vs Unusual Floods')
    plt.savefig('outputs/eda_graphs/recurring_vs_unusual.png')
    plt.close()

# worldpop stats
pop_area = load_worldpop_area(bbox)
pop_df = pd.DataFrame(list(pop_area.items()), columns=['Year', 'Total_Population'])
pop_df.set_index('Year', inplace=True)
    
# plot 3
plt.figure(figsize=(10, 6))
sns.lineplot(data=pop_df, x=pop_df.index, y='Total_Population', marker='o')
plt.title('Pop Growth')
plt.savefig('outputs/eda_graphs/population_growth.png')
plt.close()
    
# exposure
exp_list = []
for y in years:
    raster_path = f'./raw_data/worldpop/ssd_pop_{y}_CN_100m_R2025A_v1.tif'
    if os.path.exists(raster_path) and not df_floods.empty:
        floods_y = df_floods[df_floods['year'] == y]
        u_pixels = floods_y[['lat', 'lon']].drop_duplicates()
        
        exposed_pop = 0
        with rasterio.open(raster_path) as src:
            coords = [(lon, lat) for lon, lat in zip(u_pixels['lon'], u_pixels['lat'])]
            if coords:
                for val in src.sample(coords):
                    v = val[0]
                    if src.nodata is None or v != src.nodata:
                        if not np.isnan(v):
                            exposed_pop += v
        
        exp_list.append({'Year': y, 'Exposed_Population': exposed_pop})
        
if exp_list:
    exp_df = pd.DataFrame(exp_list).set_index('Year')
    
    # plot 4
    plt.figure(figsize=(10, 6))
    sns.barplot(x=exp_df.index, y=exp_df['Exposed_Population'], color='red')
    plt.title('Exposed Pop to Flooding')
    plt.savefig('outputs/eda_graphs/exposed_population.png')
    plt.close()

print("done")


