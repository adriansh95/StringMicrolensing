import pandas as pd

for v in ["v0", "v1", "v2"]:
    for i in range(67):
        parquet_dir = "/Volumes/THESIS_DATA/results/good_windows/"
        write_dir = "/Volumes/THESIS_DATA/results/good_windows2/"
        fname = f"{parquet_dir}good_windows_batch{i}_{v}.parquet"
        fname2 = f"{write_dir}good_windows_batch{i}_{v}.parquet"
        df = pd.read_parquet(fname)
        new_columns = [f"tau_{c}" for c in df.columns]
        df.columns = new_columns
        df.to_parquet(fname2)
