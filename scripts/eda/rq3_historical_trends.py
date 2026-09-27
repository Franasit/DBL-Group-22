import pandas as pd
import os
import glob
import matplotlib.pyplot as plt

def analyze_flood_trends(base_path):
    print(f"Analyzing data from {base_path}...")
    unusual_counts = {}
    recurring_counts = {}
    
    unusual_path = os.path.join(base_path, 'compact_unusual', '*.parquet')
    for file in glob.glob(unusual_path):
        year = os.path.basename(file).split('_')[-1].split('.')[0]
        try:
            df = pd.read_parquet(file)
            count = len(df)
            unusual_counts[year] = unusual_counts.get(year, 0) + count
        except Exception as e:
            print(f"Error reading {file}: {e}")
            
    recurring_path = os.path.join(base_path, 'compact_recurring', '*.parquet')
    for file in glob.glob(recurring_path):
        year = os.path.basename(file).split('_')[-1].split('.')[0]
        try:
            df = pd.read_parquet(file)
            count = len(df)
            recurring_counts[year] = recurring_counts.get(year, 0) + count
        except Exception as e:
            print(f"Error reading {file}: {e}")
            
    df_plot = pd.DataFrame({'Unusual': unusual_counts, 'Recurring': recurring_counts})
    df_plot.index = pd.to_numeric(df_plot.index)
    df_plot = df_plot.sort_index()
    
    print("\nYearly Flood Pixel/Event Counts:")
    print(df_plot)
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_plot.index, df_plot['Unusual'], marker='o', label='Unusual Floods (Severe)', color='red')
    plt.plot(df_plot.index, df_plot['Recurring'], marker='s', label='Recurring Floods (Expected)', color='blue')
    plt.title('Historical Trend of Flooding in South Sudan (2000-2025)')
    plt.xlabel('Year')
    plt.ylabel('Total Flood Events/Pixels Detected')
    plt.legend()
    plt.grid(True)
    
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    output_img = os.path.join(output_dir, 'historical_flood_trends.png')
    plt.savefig(output_img)
    print(f"\nPlot saved to {output_img}")

if __name__ == "__main__":
    base_data_path = 'raw_data/flood_masks'
    analyze_flood_trends(base_data_path)
