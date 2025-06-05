"""
This module defines RandomSampleSourcesTask.
"""
import os
from pipeline.etl_task import ETLTask

class RandomSampleSourcesTask(ETLTask):
    """
    RandomSampleSourcesTask randomly samples a subset 
    of sources without replacement and writes the resulting
    dataframe.
    """
    DEFAULT_RUN_KWARGS = {
        "extract": {
            "columns": ["id", "ra", "dec"]
        },
        "transform": {
            "seed": None,
            "sample_frac": 0.01,
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

        args : tuple
            Unused. Present for compatibility with the base 
            class method signature.

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

        Returns:
        ----------
        transformed_data : pandas.DataFrame
            The transformed data.
        """
        seed = kwargs.get("seed", self.DEFAULT_RUN_KWARGS["transform"]["seed"])
        sample_frac = kwargs.get(
            "sample_frac", self.DEFAULT_RUN_KWARGS["transform"]["sample_frac"]
        )
        result = data.sample(frac=sample_frac, random_state=seed)
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
        kwargs["iterables"] = [[None]]
        kwargs["transform"] = {
            k: user_kwargs[k]
            for k in self.DEFAULT_RUN_KWARGS["transform"] if k in user_kwargs
        }
        kwargs["extract"] = self.DEFAULT_RUN_KWARGS["extract"]
        super().run(**kwargs)

    def get_load_file_path(self, *args):
        """
        Get the load file path.

        Parameters:
        ----------
        *args: Positional arguments. These are not used by this method but are
            required to maintain compatibility with the base class interface.

        Returns:
        ----------
        load_file_path : str
            The path to the load file.
        """
        result = os.path.join(
            self.load_dir,
            "sampled_objects.parquet"
        )
        return result
