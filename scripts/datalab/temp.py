import time
from dl import queryClient as qc
from dl.helpers.utils import convert
import pandas as pd

df = pd.read_parquet("analyses/result_data/background_stats_1_to_1.parquet")
object_ids = list(df.index.get_level_values("objectid").unique())
df = pd.read_parquet(
    "/Users/adrianshestakov/Work/StringMicrolensing"
    "/analyses/result_data/bg_ids.parquet"
)
object_ids = df["id"]
oid_str = "('" + "', '".join(object_ids) + "')"

#q0 = """
#    SELECT
#        *
#    FROM
#        mydb://nsc_x_nsc_65p0
#"""
#
#q1 = f"""
#    SELECT 
#        X.*,  M.asemi, M.bsemi, M.class_star, M.dec, M.exposure, M.filter,
#        M.flags, M.kron_radius, M.mag_auto, M.magerr_auto, M.mjd, M.ra, M.theta,
#        M.x, M.y
#    FROM
#        ({q0}) AS X
#    INNER JOIN
#        nsc_dr2.meas AS M
#    ON X.id = M.objectid
#"""
#
#filtered_q1 = f"""
#    WITH derived_with_counts AS (
#        SELECT
#            *, COUNT(*) OVER (PARTITION BY id, filter) AS group_size
#        FROM ({q1}) AS derived
#    )
#    SELECT *
#    FROM derived_with_counts
#    WHERE group_size > 2
#"""
#
#q2 = f"""
#    WITH filtered_table AS (
#        {filtered_q1}
#    ),
#    valid_exposures AS (
#        SELECT DISTINCT 
#            t1_id, exposure
#        FROM
#            filtered_table
#        WHERE t1_id = id
#    )
#    SELECT
#        ft.*
#    FROM filtered_table AS ft
#    INNER JOIN
#        valid_exposures AS ve
#    ON ft.t1_id = ve.t1_id AND ft.exposure = ve.exposure
#"""
q0 = """
    SELECT id
    FROM nsc_dr2.object
    WHERE random_id
    BETWEEN 65.32 AND 65.33
    AND class_star > 0.9
"""
q1 = f"""
    SELECT M.objectid, M.class_star, M.flags, M.exposure
    FROM nsc_dr2.meas AS M
    INNER JOIN ({q0}) AS O
    ON O.id = M.objectid
"""
qc.query(
    sql=q1,
    out="mydb://sextractor_values",
    timeout=600
)
exit()
q2 = f"""
    SELECT
        COUNT(*) AS total_count
    FROM ({q1}) AS Q
"""
        #COUNT(CASE WHEN flags = 0 THEN 1 END) AS flags_0_count,
        #COUNT(CASE WHEN class_star > 0.9 THEN 1 END) AS class_star_count

#job_id = qc.query(
#    sql=q2,
#    timeout=6000,
#    fmt="pandas",
#    async_=True
#)
#print(job_id)
#df.to_parquet(
#    "/Users/adrianshestakov/Work/StringMicrolensing/"
#    "analyses/result_data/"
#    "bg_rejection_method_counts.parquet"
#)
