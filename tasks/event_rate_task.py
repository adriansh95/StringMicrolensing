"""
This module defines EventRateTask.
"""
import os
import astropy.units as u
import numpy as np
import pandas as pd
from tqdm import tqdm
from astropy.coordinates import SkyCoord
from microlensing.stringUtils import EventCalculator
from pipeline.etl_task import ETLTask

class EventRateTask(ETLTask):
    """
    EventRateTask takes a dataframe of source positions 
    and calculates the string microlensing event rate at each (ra, dec).
    """
    DEFAULT_ITERABLES = [45, 50, 55]
    DEFAULT_RUN_KWARGS = {
        "extract": {
            "columns": ["id", "ra", "dec"]
        },
        "transform": {
            "duration_bin_bounds": [1e-4, 1e4],
            "n_duration_bins": 50,
            "curly_g": 1e4,
            "host_galaxy_distance": 50,
            "host_galaxy_ra": "05h23m34s",
            "host_galaxy_dec": "-69d45.4m",
            "host_galaxy_mass": 1.38e11
        }
    }

    def transform(
            self,
            data,
            *args,
            **kwargs
        ):
        """
        Transform the data.

        Parameters:
        ----------
        data : pandas.DataFrame
            The data to transform.
        args:
            Used to maintain compatibility with base class.
        kwargs: dict
            Keyword arguments for configuring the task. This method expects the 
            following key(s):

                - duration_bin_bounds : array-like, optional
                    2 element array-like setting the upper and lower
                    limits for the duration bins. Passed to 
                    numpy.geomspace(
                        duration_bin_bounds[0],
                        duration_bin_bounds[1],
                        num=n_duration_bins
                    Default: [1e-4, 1e4].

                - n_duration_bins : int, optional
                    Number of duration bins. Passed to 
                    numpy.geomspace(
                        duration_bin_bounds[0],
                        duration_bin_bounds[1],
                        num=n_duration_bins
                    )
                    Default: 50

        Returns:
        ----------
        transformed_data : pandas.DataFrame
            The transformed data.
        """
        bins = np.geomspace(
            *kwargs["duration_bin_bounds"],
            num=kwargs["n_duration_bins"]
        ) * 86400
        event_calculator_config = {
            "curlyG": kwargs["curly_g"],
            "hostGalaxySkyCoordinates": [
                kwargs["host_galaxy_distance"] * u.kpc,
                SkyCoord(
                    ra=kwargs["host_galaxy_ra"],
                    dec=kwargs["host_galaxy_dec"]
                )
            ],
            "hostGalaxyMass": kwargs["host_galaxy_mass"] * u.solMass,
            "tensions": np.logspace(-15, -8, num=8)
        }
        distance_func = kwargs["distance_func"]
        data["source_distance"] = distance_func(data)

        def event_rate_radec(distance, ra, dec):
            ec_config = event_calculator_config.copy()
            ec_config["sourceSkyCoordinates"] = [
                distance * u.kpc,
                SkyCoord(
                    ra=ra,
                    dec=dec,
                    unit="deg",
                    frame="icrs"
                )
            ]
            event_calculator = EventCalculator(ec_config)
            event_calculator.calculate(nSteps=int(1e5))
            time_pdf, _ = event_calculator.computeLensingTimePDF(bins=bins)
            result = (
                time_pdf *
                event_calculator.results["eventRates"].reshape((-1, 1))
            ).to(1 / u.day**2).value
            return result

        result_data = np.zeros((data.shape[0] * 8, bins.shape[0]))
        result_data[:, 0] = np.tile(
            event_calculator_config["tensions"],
            data.shape[0]
        )
        result_index = pd.MultiIndex.from_product(
            [
                data["id"],
                list(range(event_calculator_config["tensions"].shape[0]))
            ],
            names=["objectid", "tension_index"]
        )
        result_columns = (
            ["tension"] + [f"duration_{i}" for i in range(bins.shape[0] - 1)]
        )

        for irow, row in tqdm(enumerate(data.itertuples(index=False))):
            result_data[8 * irow: 8 * (irow + 1), 1:] = event_rate_radec(
                row.source_distance, row.ra, row.dec
            )

        result = pd.DataFrame(
            data=result_data,
            index=result_index,
            columns=result_columns
        )
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
            "sampled_objects.parquet"
        )
        return result

    def run(self, **user_kwargs):
        """
        Run the task.

        Parameters:
        ----------
        user_kwargs: dict
            Keyword arguments for configuring the task. This method expects the 
            following key(s):
        """
        kwargs = {}
        kwargs["iterables"] = [[None]]
        kwargs["extract"] = self.DEFAULT_RUN_KWARGS["extract"].copy()
        kwargs["transform"] = self.DEFAULT_RUN_KWARGS["transform"].copy()
        kwargs["transform"].update(
            {
                k: user_kwargs[k]
                for k in self.DEFAULT_RUN_KWARGS["transform"]
                if k in user_kwargs
            }
        )

        if isinstance(user_kwargs["source_distance"], str):
            kwargs["extract"]["columns"].append(user_kwargs["source_distance"])
            kwargs["transform"]["distance_func"] = lambda x: (
                (10**((x[user_kwargs["source_distance"]] + 5) / 5))
                / 1000
            )
        elif isinstance(user_kwargs["source_distance"], (int, float)):
            kwargs["transform"]["distance_func"] = lambda _: (
                user_kwargs["source_distance"]
            )
        else:
            raise ValueError("Invalid type for 'distance_func'")

        super().run(**kwargs)

    def get_load_file_path(self, *args):
        """
        Get the load file path.

        Parameters:
        ----------
        args:
            Used to maintain compatibility with base class.
        Returns:
        ----------
        load_file_path : str
            The path to the load file.
        """
        result = os.path.join(
            self.load_dir,
            "event_rates.parquet"
        )
        return result
