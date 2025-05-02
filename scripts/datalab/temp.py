from dl import queryClient as qc

#q = """
#    SELECT objectid, mjd, filter, mag_auto, magerr_auto,
#    exposure, ccdnum, measid
#    FROM nsc_dr2.meas
#    WHERE objectid = '181013_8057'
#"""

q = """
    SELECT COUNT(id)
    FROM nsc_dr2.object
    WHERE q3c_radial_query(ra, dec, 80.8917, -69.757, 15)
    AND class_star > 0.9
"""

df = qc.query(sql=q, fmt="pandas", timeout=600)
print(df)
#df.to_parquet("/Volumes/THESIS_DATA/all_lmc_stars/all_stars.parquet")
#print(df.sort_values(by="mjd"))
