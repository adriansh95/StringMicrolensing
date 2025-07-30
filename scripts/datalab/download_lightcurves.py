import os
import re

from dl import queryClient as qc
#from dl.helpers.utils import convert
from tqdm import tqdm

def main():
    n_objects = 6.61e6
    batch_size = 50000
    n_batches = int((n_objects // batch_size) + 1)
    write_dir = "/Volumes/BACKUP_2/lightcurves/"

    for i_batch in tqdm(range(27, 132)):
        print(i_batch)
        fname = f"lightcurves_batch{i_batch}.parquet"
        id_query = f"""
            SELECT 
                id 
            FROM 
                mydb://ordered_lmc_ids
            WHERE 
                row_number BETWEEN {i_batch * batch_size}
                AND {(i_batch + 1) * batch_size - 1}
        """

        id_table_name = f"mydb://batch{i_batch}_ids"
        qc.query(
            sql=id_query,
            timeout=600,
            out=id_table_name,
            drop=True
        )
        qc.mydb_index(id_table_name, "id")

        full_query = f"""
            SELECT
                M.asemi, M.bsemi, M.class_star, M.dec, M.exposure, E.exptime,
                M.filter, M.flags, M.fwhm, M.kron_radius, M.mag_aper1,
                M.mag_aper2, M.mag_aper4, M.mag_aper8, M.mag_auto,
                M.magerr_aper1, M.magerr_aper2, M.magerr_aper4, M.magerr_aper8,
                M.magerr_auto, M.mjd,
                (M.mjd + (E.exptime / (2 * 86400))) AS mjd_mid, M.objectid,
                M.ra, E.fwhm AS seeing, M.theta, M.x, M.y
            FROM
                nsc_dr2.meas AS M
            INNER JOIN
                {id_table_name} AS IDS
            ON IDS.id = M.objectid
            INNER JOIN
                nsc_dr2.exposure AS E
            ON E.exposure = M.exposure
        """

        df = qc.query(
            sql=full_query,
            timeout=86400,
            async_=True,
            wait=True,
            verbose=True,
            poll=10,
            fmt="pandas"
        )
                
        df.to_parquet(os.path.join(write_dir, fname))
        qc.mydb_drop(id_table_name)
        print(f"Wrote {os.path.join(write_dir, fname)}")

if __name__ == "__main__":
    main()
