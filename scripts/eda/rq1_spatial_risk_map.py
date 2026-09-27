import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import sys

def compute_spatial_risk_map(base_path):
    admin_path = os.path.join(base_path, 'Administrative boundaries', 'ssd_admin1.geojson')
    admin1 = gpd.read_file(admin_path)
    
    facilities_path = os.path.join(base_path, 'health facilities', 'Sub-Saharan_public_health_facilities.geojson')
    
    try:
        gdf_facilities = gpd.read_file(facilities_path)
        
        if gdf_facilities.crs != admin1.crs:
            gdf_facilities = gdf_facilities.to_crs(admin1.crs)
            
        gdf_ssd = gpd.sjoin(gdf_facilities, admin1, how='inner', predicate='intersects')
        
        flood_file = os.path.join(base_path, 'flood_masks', 'compact_unusual', 'flood_events_h21v08_2022.parquet')
        df_flood = pd.read_parquet(flood_file)
        
        flood_file_2 = os.path.join(base_path, 'flood_masks', 'compact_unusual', 'flood_events_h20v08_2022.parquet')
        if os.path.exists(flood_file_2):
            df_flood_2 = pd.read_parquet(flood_file_2)
            df_flood = pd.concat([df_flood, df_flood_2], ignore_index=True)
        
        minx, miny, maxx, maxy = admin1.total_bounds
        
        grid_size = 0.1
        lon_bins = np.arange(minx, maxx, grid_size)
        lat_bins = np.arange(miny, maxy, grid_size)
        
        H, yedges, xedges = np.histogram2d(df_flood['lat'], df_flood['lon'], bins=[lat_bins, lon_bins])
        
        H_norm = H / H.max()
        
        def get_risk_score(geom):
            if geom is None or geom.is_empty:
                return 0
            lon, lat = geom.x, geom.y
            lat_idx = np.searchsorted(yedges, lat) - 1
            lon_idx = np.searchsorted(xedges, lon) - 1
            
            if 0 <= lat_idx < len(yedges)-1 and 0 <= lon_idx < len(xedges)-1:
                return H_norm[lat_idx, lon_idx]
            return 0
            
        gdf_ssd['Hazard_Probability'] = gdf_ssd.geometry.apply(get_risk_score)
        gdf_ssd['Exposure_Weight'] = 1.0 
        gdf_ssd['Spatial_Risk_Score'] = gdf_ssd['Hazard_Probability'] * gdf_ssd['Exposure_Weight']
        
        vulnerable = gdf_ssd.sort_values(by='Spatial_Risk_Score', ascending=False)
        print("\nTop 5 Most At-Risk Health Facilities (RQ1 Validation):")
        cols_to_print = [c for c in ['name', 'facility_name', 'type', 'adm1_name', 'Spatial_Risk_Score'] if c in vulnerable.columns]
        if not cols_to_print:
             cols_to_print = ['Hazard_Probability', 'Spatial_Risk_Score']
        print(vulnerable[cols_to_print].head(5))
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        admin1.plot(ax=ax, facecolor='lightgrey', edgecolor='black', alpha=0.5, linewidth=0.8, zorder=1)
        
        X, Y = np.meshgrid(xedges, yedges)
        H_masked = np.ma.masked_where(H_norm == 0, H_norm)
        mesh = ax.pcolormesh(X, Y, H_masked, cmap='Blues', alpha=0.6, zorder=2)
        
        high_risk = gdf_ssd[gdf_ssd['Spatial_Risk_Score'] > 0.05]
        low_risk = gdf_ssd[gdf_ssd['Spatial_Risk_Score'] <= 0.05]
        
        low_risk.plot(ax=ax, color='#2ecc71', markersize=25, edgecolor='black', linewidth=0.5, alpha=0.7, label='Low Risk Facility', zorder=3)
        high_risk.plot(ax=ax, c=high_risk['Spatial_Risk_Score'], cmap='autumn_r', markersize=120, 
                       edgecolor='black', linewidth=1.2, alpha=0.9, label='High Risk Facility', zorder=4)
        
        plt.colorbar(mesh, ax=ax, label='Flood Hazard Probability', shrink=0.5)
        
        ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, title="Facility Risk Level")
        
        plt.title('RQ1: Infrastructure Spatial Risk Map (South Sudan)', fontsize=15)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        output_img = 'outputs/rq1_infrastructure_risk_map.png'
        plt.savefig(output_img, dpi=300, bbox_inches='tight')
        
    except Exception as e:
        print(f"Error in processing: {e}")

if __name__ == "__main__":
    base_data_path = 'raw_data'
    compute_spatial_risk_map(base_data_path)
