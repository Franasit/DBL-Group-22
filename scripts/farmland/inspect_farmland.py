import rasterio
import numpy as np

file_path = "raw_data/farmland/asap_mask_crop_v04.tif"

with rasterio.open(file_path) as src:
    print("CRS:", src.crs)
    print("Width:", src.width)
    print("Height:", src.height)
    print("Bands:", src.count)
    print("Data type:", src.dtypes)
    print("Bounds:", src.bounds)
    print("Resolution:", src.res)

    data = src.read(1)

    print("Minimum value:", np.nanmin(data))
    print("Maximum value:", np.nanmax(data))

    unique_values = np.unique(data)
    print("First 30 unique values:", unique_values[:30])
    print("Number of unique values:", len(unique_values))