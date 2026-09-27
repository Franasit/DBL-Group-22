import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

def compute_spatial_risk(base_path):
    facilities_path = os.path.join(base_path, 'health facilities', 'Sub-Saharan_public_health_facilities.geojson')
    
    try:
        gdf_facilities = gpd.read_file(facilities_path)
        
        ssd_bbox = [23.5, 3.5, 36.0, 12.5] # minx, miny, maxx, maxy
        
        if 'country' in gdf_facilities.columns.str.lower():
            col_name = [c for c in gdf_facilities.columns if c.lower() == 'country'][0]
            gdf_ssd = gdf_facilities[gdf_facilities[col_name].str.contains("South Sudan", na=False, case=False)]
            if len(gdf_ssd) == 0: # fallback
                gdf_ssd = gdf_facilities.cx[ssd_bbox[0]:ssd_bbox[2], ssd_bbox[1]:ssd_bbox[3]]
        else:
            gdf_ssd = gdf_facilities.cx[ssd_bbox[0]:ssd_bbox[2], ssd_bbox[1]:ssd_bbox[3]]
            
        
        flood_file = os.path.join(base_path, 'flood_masks', 'compact_unusual', 'flood_events_h21v08_2022.parquet')
        df_flood = pd.read_parquet(flood_file)
        
        grid_size = 0.1
        lon_bins = np.arange(ssd_bbox[0], ssd_bbox[2], grid_size)
        lat_bins = np.arange(ssd_bbox[1], ssd_bbox[3], grid_size)
        
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
        cols_to_print = [c for c in ['name', 'facility_name', 'type', 'Hazard_Probability', 'Spatial_Risk_Score'] if c in vulnerable.columns]
        if not cols_to_print:
             cols_to_print = ['Hazard_Probability', 'Spatial_Risk_Score']
        print(vulnerable[cols_to_print].head(5))
        
        plt.figure(figsize=(10, 8))
        
        X, Y = np.meshgrid(xedges, yedges)
        plt.pcolormesh(X, Y, H_norm, cmap='Blues', alpha=0.6)
        
        sc = plt.scatter(gdf_ssd.geometry.x, gdf_ssd.geometry.y, 
                         c=gdf_ssd['Spatial_Risk_Score'], cmap='Reds', 
                         s=20, edgecolor='k', linewidth=0.5, zorder=5)
                         
        plt.colorbar(sc, label='Calculated Spatial Risk Score')
        plt.title('RQ1: Infrastructure Spatial Risk Score (Health Facilities vs. 2022 Floods)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        output_img = 'outputs/rq1_infrastructure_risk.png'
        plt.savefig(output_img, dpi=300)
        print(f"\nPlot saved to {output_img}")
        
    except Exception as e:
        print(f"Error in processing: {e}")

if __name__ == "__main__":
    base_data_path = 'raw_data'
    compute_spatial_risk(base_data_path)
