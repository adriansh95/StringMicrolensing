"""
This module defines BinObjectsTask.
"""
import os
import numpy as np
import healpy as hp
import pandas as pd
from utils.tasks.etl_task import ETLTask

class BinObjectsTask(ETLTask):
    """
    BinObjectsTask histograms objects
    by their (ra, dec) into bins defined by HEALPix.
    """

    def transform(self, data, healpix_nside):
        """
        Bin the objects into HEALPix bins.

        Parameters:
        ----------
        healpix_nside (int): 
        """
        phi = np.radians(data["ra"].to_numpy())
        theta = np.radians(90 - data["dec"].to_numpy())
        pixel_indices = hp.ang2pix(healpix_nside, theta, phi)
        healpix_map = np.bincount(pixel_indices)
        pixel_theta, pixel_phi = hp.pix2ang(
            healpix_nside,
            np.arange(healpix_map.shape[0])
        )
        pixel_ra = np.degrees(pixel_phi)
        pixel_dec = 90 - np.degrees(pixel_theta)
        mask = healpix_map > 0
        data = {
            "count": healpix_map[mask],
            "ra": pixel_ra[mask],
            "dec": pixel_dec[mask]
        }
        result = pd.DataFrame(data=data)
        return result

    def get_extract_file_path(self):
        """
        Get the extract file path.

        Parameters:
        ----------
        """
        result = os.path.join(
            self.extract_dir,
            "objects.parquet"
        )
        return result

    def run(self, **kwargs):
        """
        Run the task.

        Parameters:
        ----------
        kwargs : dict
            Keyword arguments for configuring the task. This method expects the 
            following key(s):

                healpix_nside: (int, optional, default: 1024): 
                    Number of sides for HEALPix grid.
                    Must be a power of 2.
        """
        healpix_nside = kwargs.get("healpix_nside", 1024)
        columns = ["ra", "dec"]
        extract_file_path = self.get_extract_file_path()
        data = self.extract(extract_file_path, columns=columns)
        transformed_data = self.transform(data, healpix_nside)
        load_file_path = self.get_load_file_path()
        self.load(transformed_data, load_file_path)

    def get_load_file_path(self):
        """
        Get the load file path.

        Parameters:
        ----------
        """
        result = os.path.join(
            self.load_dir,
            "binned_objects.parquet"
        )
        return result
