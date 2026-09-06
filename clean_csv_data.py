import os
import glob
import pandas as pd
import numpy as np

input_directories = [
    "/Users/maksadamowicz/Desktop/DBL-Group-22/Water_levels_lakes",
    "/Users/maksadamowicz/Desktop/DBL-Group-22/Darthmouth_Flood_Observatory"
]
output_base_dir = "/Users/maksadamowicz/Desktop/DBL-Group-22/Processed_Data"

def clean_data(df):
    """
    Cleans and standardizes the DataFrame.
    """
    # 1. Standardize column names
    # Lowercase, replace spaces with underscores, remove special characters
    df.columns = (
        df.columns.str.lower()
        .str.replace(' ', '_', regex=False)
        .str.replace(r'[^a-zA-Z0-9_]', '', regex=True)
    )
    
    # 2. Standardize dates
    # Guess time columns and parse them to consistent datetime format
    time_cols = [c for c in df.columns if any(x in c for x in ['date', 'time', 'year', 'month'])]
    for c in time_cols:
        try:
            # Convert to datetime (if possible) and rewrite in standardized string format
            parsed_dates = pd.to_datetime(df[c], errors='coerce')
            if parsed_dates.notna().any():
                df[c] = parsed_dates.dt.strftime('%Y-%m-%d')
        except:
            pass
            
    # 3. Handle missing values (Interpolation for time-series continuous data)
    numeric_cols = df.select_dtypes(include=['number']).columns
    for c in numeric_cols:
        # We perform a linear interpolation since discharge over time shouldn't have gaps.
        # This will fill small chunks of missing data seamlessly.
        df[c] = df[c].interpolate(method='linear')
        
        # If there are any missing values left (e.g., at the very start/end of the file), 
        # we can backfill/forward fill them as a fallback.
        df[c] = df[c].bfill().ffill()
        
    return df

if __name__ == "__main__":
    print(f"Starting data cleaning process...\n")
    
    for d in input_directories:
        if not os.path.exists(d):
            print(f"Skipping (Directory not found): {d}")
            continue
            
        csv_files = glob.glob(os.path.join(d, "**", "*.csv"), recursive=True)
        print(f"Found {len(csv_files)} files in {os.path.basename(d)}.")
        
        for file_path in csv_files:
            try:
                # Read the original file
                df = pd.read_csv(file_path, low_memory=False)
                
                # Apply transformations
                df_cleaned = clean_data(df)
                
                # Construct output path keeping the original folder structure
                rel_path = os.path.relpath(file_path, start=os.path.dirname(d)) 
                output_path = os.path.join(output_base_dir, rel_path)
                
                # Create directories if they don't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Save the processed data
                df_cleaned.to_csv(output_path, index=False)
                print(f"  -> Cleaned and saved: {output_path}")
                
            except Exception as e:
                print(f"  -> Error processing {file_path}: {e}")
                
    print(f"\nAll cleaning finished. Processed files are in: {output_base_dir}")
