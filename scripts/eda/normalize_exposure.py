import pandas as pd
import numpy as np

# load aligned data
df = pd.read_csv("../../outputs/adm3_exposure_aligned.csv")

# cols to normalize
exposure_cols = ['population_2024', 'cattle_sum', 'cropland_score', 'health_facilities_count']
hazard_col = 'flood_hazard_2022_count'

# min-max scale
def min_max_scale(series):
    if series.max() == series.min():
        return np.zeros(len(series))
    return (series - series.min()) / (series.max() - series.min())

# apply
for col in exposure_cols:
    df[f"{col}_norm"] = min_max_scale(df[col])
    
df[f"{hazard_col}_norm"] = min_max_scale(df[hazard_col])

# get combined exposure (equal weights)
norm_cols = [f"{col}_norm" for col in exposure_cols]
df['combined_exposure_score'] = df[norm_cols].mean(axis=1)

# sort
df_sorted = df.sort_values('combined_exposure_score', ascending=False)

# save
output_path = "../../outputs/adm3_exposure_normalized.csv"
df.to_csv(output_path, index=False)
print(f"saved {output_path}")

