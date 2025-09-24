import requests
import pandas as pd

API_KEY = "579b464db66ec23bdd00000126ae4247eb1e4b8a4e428236d2b1a905"
RESOURCE_ID = "35985678-0d79-46b4-9ed6-6f13308a1d24"

def fetch_data(date_str):
    url = (
        f"https://api.data.gov.in/resource/{RESOURCE_ID}"
        f"?api-key={API_KEY}&format=json&limit=10000&filters[Arrival_Date]={date_str}"
    )
    response = requests.get(url)
    response.raise_for_status()
    records = response.json().get("records", [])
    if not records:
        print(f"No data found for {date_str}.")
        return 0
    df = pd.DataFrame(records)
    filename = f"daily_{date_str}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved {filename} with {len(df)} rows.")
    return len(df)

if __name__ == "__main__":
    from datetime import datetime, timedelta
    # Kal ki date (yesterday's date) in YYYY-MM-DD format
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    fetch_data(yesterday)
