from dl import queryClient as qc

#q = """
#    SELECT objectid, mjd, filter, mag_auto, magerr_auto,
#    exposure, ccdnum, measid
#    FROM nsc_dr2.meas
#    WHERE objectid = '181013_8057'
#"""

q = """
    SELECT id, ra, dec
    FROM nsc_dr2.object
    WHERE q3c_radial_query(ra, dec, 80.8917, -69.757, 5.5)
    AND class_star > 0.9
"""

df = qc.query(sql=q, fmt="pandas", timeout=600)
df.to_parquet("/Volumes/THESIS_DATA/all_lmc_stars/all_stars.parquet")
#print(df.sort_values(by="mjd"))
