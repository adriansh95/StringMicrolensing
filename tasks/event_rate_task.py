"""
This module defines EventRateTask.
"""
import os
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from config.efficiency_config import tau_bins
from microlensing.stringUtils import EventCalculator
from pipeline.etl_task import ETLTask
from tqdm import tqdm

class EventRateTask(ETLTask):
    """
    EventRateTask takes a dataframe of (n_objects, ra, dec) and calculates
    the string microlensing event rate at each (ra, dec) and weights by
    n_objects.
    """
    def transform(self, data, source_distance):
        """
        Transform the data.

        Parameters:
        ----------
        data : pandas.DataFrame
            The data to transform.
        source_distance : astropy.units.Quantity
            Distance at which to place the sources.

        Returns:
        ----------
        transformed_data : pandas.DataFrame
            The transformed data.
        """
        bins = tau_bins * 86400
        event_calculator_config = {
            "curlyG": 1e4,
            "hostGalaxySkyCoordinates": [
                50 * u.kpc,
                SkyCoord(ra="05h23m34s", dec="69d45.4m", frame="icrs")
            ],
            "hostGalaxyMass": 1.38e11 * u.solMass,
            "tensions": np.logspace(-15, -8, num=8)
        }
        result_data = np.zeros(
            (
                bins.shape[0] - 1,
                event_calculator_config["tensions"].shape[0]
            )
        )

        for row in tqdm(data.itertuples(index=False)):
            event_calculator_config["sourceSkyCoordinates"] = [
                source_distance * u.kpc,
                SkyCoord(ra=row.ra, dec=row.dec, unit="deg", frame="icrs")
            ]
            event_calculator = EventCalculator(event_calculator_config)
            event_calculator.calculate()
            time_pdf, _ = event_calculator.computeLensingTimePDF(bins=bins)
            result_data += (
                time_pdf.transpose() * row.count * (
                    event_calculator
                    .results["eventRates"]
                )
            ).to(1 / u.day**2).value

        result_column_names = [f"mu_{i}" for i in range(-15, -7)]
        result = pd.DataFrame(data=result_data, columns=result_column_names)
        return result

    def get_extract_file_path(self, *args):
        """
        Get the extract file path.

        Parameters:
        ----------
        Args:
        *args: Positional arguments. These are not used by this method but are
            required to maintain compatibility with the base class interface.

        Returns:
        ----------
        extract_file_path : str
            The path to the extract file.
        """
        result = os.path.join(
            self.extract_dir,
            "binned_objects.parquet"
        )
        return result

    def run(self, **kwargs):
        """
        Run the task.

        Parameters:
        ----------
        kwargs: dict
            Keyword arguments for configuring the task. This method expects the 
            following key(s):
                source_distances : array-like:
                    Distance (in kpc) at which to place the sources.
                    Default: [42, 50, 58].
        """
        source_distances = kwargs.get("source_distances", [42, 50, 58])
        kwargs["iterables"] = [source_distances]
        super().run(**kwargs)

    def get_load_file_path(self, source_distance):
        """
        Get the load file path.

        Parameters:
        ----------
        source_distance (astropy.units.Quantity):
            Distance at which to place the sources.

        Returns:
        ----------
        load_file_path : str
            The path to the load file.
        """
        result = os.path.join(
            self.load_dir,
            f"event_rates_{source_distance}.parquet"
        )
        return result
