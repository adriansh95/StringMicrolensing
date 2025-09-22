import pandas as pd
import os
import glob

from tqdm import tqdm

def main():
    lightcurve_dir = (
        "/Volumes/THESIS1/results/efficiency/"
        "lensed_lightcurves/24_7_2025/"
    )
    write_dir = os.path.join(lightcurve_dir, "concatted")
    batch_lc_dir = os.path.join(lightcurve_dir, "batch")

    for i_tau in tqdm(range(0, 49)):
        globstr = f"lensed_lightcurves*duration{i_tau}.parquet"
        lightcurve_files = glob.glob(os.path.join(batch_lc_dir, globstr))

        if len(lightcurve_files) > 0:
            dfs = [
                pd.read_parquet(f) for f in lightcurve_files
            ]
            df = pd.concat(dfs, axis=0)
            write_name = os.path.join(
                write_dir,
                f"lensed_lightcurves_duration{i_tau}.parquet"
            )
            df.to_parquet(write_name)
            print(f"Wrote {write_name}")

if __name__ == "__main__":
    main()
