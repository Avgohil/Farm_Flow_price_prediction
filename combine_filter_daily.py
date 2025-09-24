

import pandas as pd
import glob
import os

# List of 21 commodities to keep
desired_commodities = [
    'Maize', 'Onion', 'Soyabean', 'Wheat', 'Banana', 'Coconut',
    'Green Chilli', 'Potato', 'Tomato', 'Bajra(Pearl Millet/Cumbu)',
    'Rice', 'Jowar(Sorghum)', 'Groundnut', 'Mustard', 'Grapes',
    'Cotton', 'Sugarcane', 'Cummin Seed(Jeera)', 'Arecanut(Betelnut/Supari)',
    'Ragi (Finger Millet)', 'Jute', 'Sugarcane'
]

# Mapping from API columns to final columns
col_map = {
    'State': 'state_name',
    'District': 'district_name',
    'Market': 'market_name',
    'Commodity': 'commodity_name',
    'Variety': 'variety',
    'Min Price': 'min_price',
    'Max Price': 'max_price',
    'Modal Price': 'modal_price',
    'Arrival Date': 'date'
}

# Dynamically find all daily_YYYY-MM-DD.csv files in the current directory
files = sorted(glob.glob('daily_20*.csv'))
print(f"Found files: {files}")

df_list = []
for file in files:
    df = pd.read_csv(file, dtype=str)
    # Make all columns lower and replace spaces/underscores for robust renaming
    df.columns = [c.strip().lower().replace(' ', '_').replace('-', '_') for c in df.columns]
    # Robust mapping
    robust_map = {
        'state': 'state_name',
        'district': 'district_name',
        'market': 'market_name',
        'commodity': 'commodity_name',
        'variety': 'variety',
        'min_price': 'min_price',
        'max_price': 'max_price',
        'modal_price': 'modal_price',
        'arrival_date': 'date'
    }
    df = df.rename(columns=robust_map)
    print(f"Columns after renaming for {file}: {list(df.columns)}")
    # Drop extra columns if present
    for col in ['grade', 'state']:
        if col in df.columns:
            df = df.drop(columns=col)
    df_list.append(df)

# Combine all data
if df_list:
    combined = pd.concat(df_list, ignore_index=True)
else:
    combined = pd.DataFrame()

# Filter for desired commodities
filtered = combined[combined['commodity_name'].isin(desired_commodities)] if not combined.empty else combined

# Keep only the required columns
final_cols = [
    'state_name', 'district_name', 'market_name', 'commodity_name', 'variety',
    'min_price', 'max_price', 'modal_price', 'date'
]
filtered = filtered[final_cols] if not filtered.empty else filtered

# Remove rows already present in master_data_2019_2025.csv
master_file = 'master_data_2019_2025.csv'

if os.path.exists(master_file):
    master_df = pd.read_csv(master_file, dtype=str)
    key_cols = ['state_name', 'district_name', 'market_name', 'commodity_name', 'variety', 'date']
    # Merge to find only new rows
    merged = filtered.merge(master_df[key_cols], on=key_cols, how='left', indicator=True)
    filtered = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

# Always overwrite daily_new.csv with only new data
with open('daily_new.csv', 'w') as f:
    if not filtered.empty:
        filtered.to_csv(f, index=False)
    else:
        # Write only header if no new data
        f.write(','.join(final_cols) + '\n')

print(f"Rows after filtering and removing old data: {len(filtered)}")
print(f"Columns: {list(filtered.columns)}")
