import os
import re

from dl import queryClient as qc
#from dl.helpers.utils import convert
from tqdm import tqdm

def main():
    n_objects = 6.61e6
    #batch_size = 50000
    #n_batches = int((n_objects // batch_size) + 1)
    n_batches = 1
#    write_dir = (
#        "/VOLUMES/THESIS1/lightcurves/"
#    )
    write_dir = "demo/"

    for i_batch in tqdm(range(0, n_batches)):
        print(i_batch)
        fname = f"lightcurves.parquet"
        #id_query = f"""
        #    SELECT 
        #        id 
        #    FROM 
        #        mydb://ordered_lmc_ids
        #    WHERE 
        #        row_number BETWEEN {i_batch * batch_size}
        #        AND {(i_batch + 1) * batch_size - 1}
        #"""
        id_query = f"""
            SELECT 
                id 
            FROM 
                mydb://ordered_lmc_ids
            ORDER BY RANDOM()
            LIMIT 1000
        """

#        id_table_name = f"mydb://batch{i_batch}_ids"
        id_table_name = "mydb://temp"
        qc.query(
            sql=id_query,
            timeout=600,
            out=id_table_name,
            drop=True
        )
        qc.mydb_index(id_table_name, "id")
        mydb_query = f"SELECT id FROM {id_table_name}"
        object_query = f"""
            WITH IDS AS ({mydb_query})
            SELECT 
                O.id, O.ra, O.dec
            FROM 
                nsc_dr2.object AS O
            INNER JOIN IDS ON O.id = IDS.id
        """
        objects = qc.query(sql=object_query, timeout=600, fmt="pandas")
        objects.to_parquet("demo/objects/sampled_objects.parquet")

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
                ({mydb_query}) AS IDS
            ON IDS.id = M.objectid
            INNER JOIN
                nsc_dr2.exposure AS E
            ON E.exposure = M.exposure
        """

#        df = qc.query(
#            sql=full_query,
#            async_=True,
#            wait=True,
#            poll=10,
#            verbose=True,
#            timeout=86400,
#            fmt="pandas"
#        )
        df = qc.query(sql=full_query, timeout=600, fmt="pandas")
        df.to_parquet("demo/lightcurves/all_lightcurves.parquet")
                
        #df.to_parquet(os.path.join(write_dir, fname))
        qc.mydb_drop(id_table_name)
        #print(f"Wrote {os.path.join(write_dir, fname)}")

if __name__ == "__main__":
    main()
