import os
import requests
import pandas as pd

API_KEY = "579b464db66ec23bdd00000126ae4247eb1e4b8a4e428236d2b1a905"
RESOURCE_ID = "35985678-0d79-46b4-9ed6-6f13308a1d24"
MASTER_FILE = "master_data_2019_2025.csv"


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


def infer_date_column(df_cols):
    # prefer lowercase 'date', but accept common variants
    candidates = [c for c in df_cols if c.lower() in ("date", "arrival_date", "arrivaldate")]
    return candidates[0] if candidates else None


if __name__ == "__main__":
    from datetime import datetime, timedelta

    yesterday_dt = datetime.now() - timedelta(days=1)
    yesterday = yesterday_dt.strftime("%Y-%m-%d")

    # Determine the start date: one day after the last date present in master file
    start_dt = None

    if os.path.exists(MASTER_FILE):
        try:
            df_master = pd.read_csv(MASTER_FILE, dtype=str)
            date_col = infer_date_column(df_master.columns)
            if date_col is not None:
                # Robust parsing: try multiple strategies and ignore unparsable rows
                def robust_parse_series(ser):
                    ser = ser.fillna("").astype(str).str.strip()
                    # 1) try fast vectorized parse with infer and dayfirst=False
                    parsed = pd.to_datetime(ser, errors='coerce', infer_datetime_format=True, dayfirst=False)
                    # 2) for missing, try dayfirst=True
                    mask = parsed.isna()
                    if mask.any():
                        parsed_dayfirst = pd.to_datetime(ser[mask], errors='coerce', infer_datetime_format=True, dayfirst=True)
                        parsed.loc[mask] = parsed_dayfirst

                    # 3) custom formats fallback for remaining unparsable values
                    remaining_mask = parsed.isna()
                    if remaining_mask.any():
                        from datetime import datetime
                        fmts = [
                            "%Y-%m-%d",
                            "%d-%m-%Y",
                            "%d/%m/%Y",
                            "%m/%d/%Y",
                            "%d/%m/%y",
                            "%d-%b-%Y",
                            "%d %b %Y",
                            "%d %B %Y",
                            "%Y/%m/%d",
                            "%m-%d-%Y",
                        ]
                        for idx, val in ser[remaining_mask].items():
                            if not val:
                                continue
                            parsed_val = None
                            # remove common suffixes and time parts
                            v = val.split()[0]
                            v = v.replace('.', '/')
                            for fmt in fmts:
                                try:
                                    dt = datetime.strptime(v, fmt)
                                    parsed_val = pd.Timestamp(dt)
                                    break
                                except Exception:
                                    continue
                            parsed.at[idx] = parsed_val

                    return parsed

                parsed = robust_parse_series(df_master[date_col])
                # Ignore unparsable rows
                valid = parsed.dropna()
                if not valid.empty:
                    last_date = valid.max()
                    start_dt = last_date + timedelta(days=1)
                    print(f"Found last valid date in {MASTER_FILE}: {last_date.date()}; will fetch from {start_dt.date()} to {yesterday}.")
                else:
                    print(f"No valid dates parsed from {MASTER_FILE}.{date_col}. Falling back to fetching yesterday only.")
            else:
                print(f"No date column found in {MASTER_FILE}. Falling back to fetching yesterday only.")
        except Exception as e:
            print(f"Error reading {MASTER_FILE}: {e}. Falling back to fetching yesterday only.")
    else:
        print(f"{MASTER_FILE} not found. Fetching yesterday only ({yesterday}).")

    if start_dt is None:
        start_dt = yesterday_dt

    # If start is after yesterday, nothing to fetch
    if start_dt.date() > yesterday_dt.date():
        print("Master is up to date. No new dates to fetch.")
    else:
        cur = start_dt
        fetched_total = 0
        while cur.date() <= yesterday_dt.date():
            date_str = cur.strftime("%Y-%m-%d")
            try:
                fetched_total += fetch_data(date_str)
            except Exception as e:
                print(f"Error fetching {date_str}: {e}")
            cur = cur + timedelta(days=1)

        print(f"Finished fetching. Total new rows retrieved (approx): {fetched_total}")
