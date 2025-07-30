import pandas as pd
from tqdm import tqdm

summary_table = pd.read_parquet(
    "/Volumes/THESIS_DATA/results/summary_table/summary_table.parquet",
    columns=["root_2_label"]
)
summary_table = summary_table.query("root_2_label == 'background'")
batches = summary_table.index.get_level_values("batch_number").unique()

for batch in tqdm(batches[19:]):
    print(batch)
    data = pd.read_parquet(
        "/Volumes/THESIS_DATA/results/kde_labelled_lightcurves/"
        f"kde_labelled_lightcurves_batch{batch}.parquet"
    )
    mask = data["objectid"].isin(
        summary_table.xs(batch, level="batch_number").index
    )
    data = data.loc[mask]
    data.to_parquet(
        "/Users/adrianshestakov/Work/StringMicrolensing/analyses/"
        "result_data/22_7_2025_background_data/"
        f"background_data_batch{batch}.parquet"
    )
