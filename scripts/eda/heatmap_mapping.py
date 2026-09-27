import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
raw_data_dir = PROJECT_ROOT / 'raw_data' if (PROJECT_ROOT / 'raw_data').exists() else PROJECT_ROOT / 'data-JBG060-2026'

df = pd.read_csv(PROJECT_ROOT / 'outputs' / 'adm3_exposure_normalized.csv')
gdf = gpd.read_file(raw_data_dir / 'Administrative boundaries' / 'ssd_admin3.geojson')

# merge
merged = gdf.merge(df[['adm3_pcode', 'combined_exposure_score', 'flood_hazard_2022_count_norm']], on='adm3_pcode')
merged['extreme_score'] = merged['combined_exposure_score'] + merged['flood_hazard_2022_count_norm']

# get top 3 for both metrics to label them
top3_hazard = merged.nlargest(3, 'flood_hazard_2022_count_norm')
top3_exposure = merged.nlargest(3, 'combined_exposure_score')
top_points = pd.concat([top3_hazard, top3_exposure]).drop_duplicates(subset=['adm3_pcode'])

# revert to ylorrd
cmap = 'YlOrRd'

# scale span full range
vmax_val = merged['extreme_score'].max()
norm = colors.Normalize(vmin=0, vmax=vmax_val)

# plot scatter
plt.figure(figsize=(9, 7))
sc = plt.scatter(
    merged['combined_exposure_score'], 
    merged['flood_hazard_2022_count_norm'], 
    c=merged['extreme_score'], 
    cmap=cmap,
    norm=norm,
    alpha=0.9, 
    edgecolor='black',
    linewidth=0.5
)

# label the points
for idx, row in top_points.iterrows():
    name = row['adm3_name']
    
    # fix overlapping labels for geger and northern bari
    if name == 'Geger':
        xytext = (-25, -20)  # below left
    elif name == 'Northern Bari':
        xytext = (-25, 20)   # above left
    elif name == 'Bagari':
        xytext = (-25, 20)
    else:
        xytext = (6, 6)
        
    # draw arrow if offset
    arrowprops = dict(arrowstyle='-', color='gray', lw=0.8) if xytext != (6, 6) else None
    
    plt.annotate(
        name, 
        (row['combined_exposure_score'], row['flood_hazard_2022_count_norm']),
        xytext=xytext, 
        textcoords='offset points', 
        fontsize=9, 
        fontweight='bold',
        color='black',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5),
        arrowprops=arrowprops
    )

# standard colorbar without max extension
plt.colorbar(sc, label='Combined Extreme Score')
plt.xlabel('Exposure Index')
plt.ylabel('Flood Hazard Score (2022)')
plt.title('Scatter Plot: Exposure vs Hazard')
plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'outputs' / 'exposure_scatter_plot.png', dpi=300)
plt.close()

# plot heatmap
fig, ax = plt.subplots(figsize=(12, 10))

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array(merged['extreme_score'])

merged.plot(
    column='extreme_score', 
    cmap=cmap, 
    norm=norm,
    edgecolor='grey', 
    linewidth=0.4, 
    ax=ax
)

cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', shrink=0.6, pad=0.02)
cbar.set_label('Combined Extreme Risk Score (Hazard + Exposure)', fontsize=11)

ax.axis('off')
ax.set_title('Risk Heatmap: YlOrRd Scale', fontsize=15)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'outputs' / 'exposure_heatmap_map.png', dpi=300)
plt.close()

