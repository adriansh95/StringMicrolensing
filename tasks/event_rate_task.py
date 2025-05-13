"""
This module defines EventRateTask.
"""
import os
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from tqdm import tqdm
from microlensing.stringUtils import EventCalculator
from pipeline.etl_task import ETLTask

class EventRateTask(ETLTask):
    """
    EventRateTask takes a dataframe of source positions 
    randomly samples a subset without replacement and calculates
    the string microlensing event rate at each (ra, dec).
    """
    DEFAULT_ITERABLES = [45, 50, 55]
    DEFAULT_RUN_KWARGS = {
        "extract": {
            "columns": ["ra", "dec"]
        },
        "transform": {
            "seed": None,
            "sample_frac": 0.01,
            "tau_bin_bounds": [1e-4, 1e4],
            "n_tau_bins": 50
        }
    }

    def transform(
            self,
            data,
            source_distance,
            **kwargs
        ):
        """
        Transform the data.

        Parameters:
        ----------
        data : pandas.DataFrame
            The data to transform.
        source_distance : astropy.units.Quantity
            Distance at which to place the sources.
        kwargs: dict
            Keyword arguments for configuring the task. This method expects the 
            following key(s):
                - seed : int, array-like, BitGenerator, np.random.RandomState,
                         np.random.Generator, optional

                    Seed used to initialize the random number generator,
                    passed to `pandas.DataFrame.sample` as the `random_state`
                    argument. Controls the reproducibility of sampling.

                - sample_frac : float, optional
                    Fraction of the data to sample, passed to 
                    `pandas.DataFrame.sample` as the `frac` argument.
                    Must be between 0 and 1.

                - tau_bins : numpy.array, optional
                    Bins for the event rate duration probability density
                    function. The default value is taken from 

        Returns:
        ----------
        transformed_data : pandas.DataFrame
            The transformed data.
        """
        seed = kwargs.get("seed", self.DEFAULT_RUN_KWARGS["transform"]["seed"])
        sample_frac = kwargs.get(
            "sample_frac", self.DEFAULT_RUN_KWARGS["transform"]["sample_frac"]
        )
        tau_bin_bounds = kwargs.get(
            "tau_bin_bounds",
            self.DEFAULT_RUN_KWARGS["transform"]["tau_bin_bounds"]
        )
        n_tau_bins = kwargs.get(
            "n_tau_bins",
            self.DEFAULT_RUN_KWARGS["transform"]["n_tau_bins"]
        )
        bins = np.geomspace(
            tau_bin_bounds[0],
            tau_bin_bounds[1],
            num=n_tau_bins
        ) * 86400
        # bounds are INCLUSIVE on both sides
        log_tension_bounds = (-15, -8)
        event_calculator_config = {
            "curlyG": 1e4,
            "hostGalaxySkyCoordinates": [
                50 * u.kpc,
                SkyCoord(ra="05h23m34s", dec="-69d45.4m", frame="icrs")
            ],
            "hostGalaxyMass": 1.38e11 * u.solMass,
            "tensions": np.logspace(
                log_tension_bounds[0],
                log_tension_bounds[1],
                num=(log_tension_bounds[1] - log_tension_bounds[0] + 1)
            )
        }
        result_data = np.zeros(
            (
                bins.shape[0] - 1,
                event_calculator_config["tensions"].shape[0]
            )
        )
        sampled_data = data.sample(frac=sample_frac, random_state=seed)

        for row in tqdm(sampled_data.itertuples(index=False)):
            event_calculator_config["sourceSkyCoordinates"] = [
                source_distance * u.kpc,
                SkyCoord(ra=row.ra, dec=row.dec, unit="deg", frame="icrs")
            ]
            event_calculator = EventCalculator(event_calculator_config)
            event_calculator.calculate(nSteps=int(1e6))
            time_pdf, _ = event_calculator.computeLensingTimePDF(bins=bins)

            result_data += (
                time_pdf.transpose() * (
                    event_calculator
                    .results["eventRates"]
                )
            ).to(1 / u.day**2).value

        result_data *= (data.shape[0] / sampled_data.shape[0])
        result = pd.DataFrame(
            data=result_data,
            columns=[
                f"mu_{i}"
                for i in range(log_tension_bounds[0], log_tension_bounds[1] + 1)
            ]
        )
        result.index.name = "tau_index"
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
            "objects.parquet"
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
                - source_distances : array-like:
                    Distance (in kpc) at which to place the sources.
                    Default: [45, 50, 55].

                - seed : int, array-like, BitGenerator, np.random.RandomState,
                         np.random.Generator, optional

                    Seed used to initialize the random number generator,
                    passed to `pandas.DataFrame.sample` as the `random_state`
                    argument. Controls the reproducibility of sampling.

                - sample_frac : float, optional
                    Fraction of the data to sample, passed to 
                    `pandas.DataFrame.sample` as the `frac` argument.
                    Must be between 0 and 1.
        """
        kwargs = {}
        kwargs["iterables"] = [
            user_kwargs.pop("source_distances", self.DEFAULT_ITERABLES)
        ]
        kwargs["transform"] = {
            k: user_kwargs[k]
            for k in self.DEFAULT_RUN_KWARGS["transform"] if k in user_kwargs
        }
        kwargs["extract"] = self.DEFAULT_RUN_KWARGS["extract"]
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
