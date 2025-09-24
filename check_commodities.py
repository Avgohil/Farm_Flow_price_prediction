import pandas as pd

df = pd.read_csv("daily_new.csv")
print("Unique commodities in daily_new.csv:")
print(df['commodity_name'].unique())
print(f"Total rows: {len(df)}")
