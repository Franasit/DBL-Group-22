import os
import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')

import os
import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')

base_data_path = '../../raw_data'
output_dir = '../../outputs'
os.makedirs(output_dir, exist_ok=True)

# load adm3
adm3_path = os.path.join(base_data_path, 'Administrative boundaries', 'ssd_admin3.geojson')
gdf_adm3 = gpd.read_file(adm3_path)
gdf_adm3 = gdf_adm3.to_crs("EPSG:4326")

# population
pop_path = os.path.join(base_data_path, 'worldpop', 'ssd_pop_2024_CN_100m_R2025A_v1.tif')
if os.path.exists(pop_path):
    stats = zonal_stats(gdf_adm3, pop_path, stats="sum")
    gdf_adm3['population_2024'] = [s['sum'] if s['sum'] else 0 for s in stats]

# livestock
cattle_path = os.path.join(base_data_path, 'farmland', 'geonode__cattle_gha.tif')
if os.path.exists(cattle_path):
    stats = zonal_stats(gdf_adm3, cattle_path, stats="sum")
    gdf_adm3['cattle_sum'] = [max(0, s['sum']) if s['sum'] else 0 for s in stats]

# cropland mask
crop_path = os.path.join(base_data_path, 'farmland', 'asap_mask_crop_v04.tif')
if os.path.exists(crop_path):
    stats = zonal_stats(gdf_adm3, crop_path, stats="sum", nodata=255)
    gdf_adm3['cropland_score'] = [s['sum'] if s['sum'] else 0 for s in stats]

# health facilities
health_path = os.path.join(base_data_path, 'health facilities', 'Sub-Saharan_public_health_facilities.geojson')
if os.path.exists(health_path):
    gdf_health = gpd.read_file(health_path).to_crs("EPSG:4326")
    if 'Country' in gdf_health.columns:
        gdf_health = gdf_health[gdf_health['Country'].str.contains("South Sudan", na=False)]
    
    joined = gpd.sjoin(gdf_health, gdf_adm3, how="inner", predicate="intersects")
    counts = joined.groupby('index_right').size()
    gdf_adm3['health_facilities_count'] = gdf_adm3.index.map(counts).fillna(0)

# flood hazard 2022
flood_files = [
    os.path.join(base_data_path, 'flood_masks', 'compact_unusual', 'flood_events_h21v08_2022.parquet'),
    os.path.join(base_data_path, 'flood_masks', 'compact_unusual', 'flood_events_h20v08_2022.parquet')
]

flood_dfs = []
for f in flood_files:
    if os.path.exists(f):
        flood_dfs.append(pd.read_parquet(f))
        
if flood_dfs:
    df_flood = pd.concat(flood_dfs, ignore_index=True)
    gdf_flood = gpd.GeoDataFrame(df_flood, geometry=gpd.points_from_xy(df_flood.lon, df_flood.lat), crs="EPSG:4326")
    
    joined_flood = gpd.sjoin(gdf_flood, gdf_adm3, how="inner", predicate="intersects")
    flood_counts = joined_flood.groupby('index_right').size()
    gdf_adm3['flood_hazard_2022_count'] = gdf_adm3.index.map(flood_counts).fillna(0)

df_adm3 = pd.DataFrame(gdf_adm3.drop(columns='geometry'))
csv_file = os.path.join(output_dir, 'adm3_exposure_aligned.csv')
df_adm3.to_csv(csv_file, index=False)
print(f"saved to {csv_file}")
