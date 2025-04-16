import os
import time
import logging
logger = logging.getLogger(__name__)

from dl import queryClient as qc
from dl.helpers.utils import convert
from tqdm import tqdm

def main():
    log_dir = "/Users/adrianshestakov/Work/stringScratch/logs"
    log_name = os.path.join(log_dir, "download_lightcurves.log")
    logging.basicConfig(filename=log_name, level=logging.INFO)
    n_batches = 67
    batch_size = 100000
    write_dir = "/Volumes/Thesis_data/"
    qc.mydb_drop("temp_ids")

    for i_batch in tqdm(range(41, 42)):
        fname = f"lightcurves_batch{i_batch}.parquet"
        sq = f"""SELECT id FROM mydb://ordered_lmc_ids
                 WHERE row_number BETWEEN {i_batch * batch_size}
                 AND {(i_batch + 1) * batch_size - 1}"""
        qc.query(sql=sq, out="mydb://temp_ids")
        qc.mydb_index("temp_ids", "id")
        q = f"""SELECT m.objectid, m.filter, m.mag_auto, 
                       m.magerr_auto, m.mjd, e.exptime,
                       (m.mjd + (e.exptime / (2 * 86400))) AS mjd_mid
                FROM nsc_dr2.meas AS m
                INNER JOIN mydb://temp_ids AS t
                ON t.id = m.objectid
                INNER JOIN nsc_dr2.exposure AS e
                ON e.exposure = m.exposure"""

        job_id = qc.query(sql=q, timeout=2400, async_=True)

        while qc.status(job_id) == "EXECUTING":
            time.sleep(10)

        if qc.status(job_id) == "COMPLETED":
            df = convert(qc.results(job_id))
            df.to_parquet(os.path.join(write_dir, fname))
            logger.info(f"Wrote batch {i_batch}")
        elif qc.status(job_id) == "ERROR":
            e = qc.error(job_id)
            logging.error(f"Error batch {i_batch}: {e}")
        else:
            s = f"Error batch {i_batch}: Something unexpected ocurred."
            logging.error(s)
        qc.mydb_drop("temp_ids")

if __name__ == "__main__":
    main()
