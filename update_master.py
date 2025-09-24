import pandas as pd

def update_master(master_file, daily_file):
    # Load both files
    df_master = pd.read_csv(master_file, dtype=str)
    df_daily = pd.read_csv(daily_file, dtype=str)

    key_cols = ['state_name', 'district_name', 'market_name', 'commodity_name', 'variety', 'date']

    # Find truly new rows in daily that are not already in master
    merged = df_daily.merge(df_master[key_cols], on=key_cols, how='left', indicator=True)
    new_daily_rows = merged[merged['_merge'] == 'left_only']
    num_new_rows = len(new_daily_rows)

    # Append only new rows to master
    if num_new_rows > 0:
        df_master = pd.concat([df_master, df_daily.loc[new_daily_rows.index]], ignore_index=True)
        # Deduplicate just in case
        df_master = df_master.drop_duplicates(subset=key_cols, keep='last').reset_index(drop=True)
        df_master.to_csv(master_file, index=False)
    else:
        # Still deduplicate in case of manual edits
        df_master = df_master.drop_duplicates(subset=key_cols, keep='last').reset_index(drop=True)
        df_master.to_csv(master_file, index=False)

    print(f"Number of new rows added: {num_new_rows}")
    print(f"Final total number of rows in {master_file}: {len(df_master)}")

if __name__ == "__main__":
    update_master('master_data_2019_2025.csv', 'daily_new.csv')
