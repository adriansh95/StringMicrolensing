"""
This module defines ETLTask, the Base task defining methods and
abstract methods which subclasses will inherit and must implement.
"""

from itertools import product
from abc import ABC, abstractmethod
from tqdm import tqdm
import pandas as pd

class ETLTask(ABC):
    """
    Base task defining methods and abstract methods which 
    subclasses will inherit and must implement.

    Attributes:
        extract_dir (str): Directory containing the input data files to be
                           processed.
        load_dir (str): Directory where the transformed data files will be
                        written.
    Methods:
        extract (data_file_path: str, **kwargs: Any) -> (pandas.DataFrame):
            Read in the data.
        transform
            (data: pandas.DataFrame, **kwargs: Any) -> (pandas.DataFrame):
            Transform the data.
        load (data: pandas.DataFrame, data_file_path: str): Load the
            transformed data.
        run (**kwargs: Any):
            Run the task.
    """
    def __init__(self, extract_dir, load_dir):
        self.extract_dir = extract_dir
        self.load_dir = load_dir

    def extract(self, data_file_path, **kwargs):
        """
        A wrapper for pandas.read_parquet.

        Parameters
        ----------
        data_file_path: (str)
            The file to read in.
        **kwargs: Any
            Additional keyword arguments passed to `pandas.read_parquet`
            For example:
                - engine (str): Parquet library to use
                    ('auto', 'pyarrow', 'fastparquet').
                - columns (list[str], optional): List of columns to read
                    from the file.
                - storage_options (dict, optional): Extra options for remote
                    file systems.
        """
        result = pd.read_parquet(data_file_path, **kwargs)
        return result

    @abstractmethod
    def transform(self, data):
        """
        Abstract method to transform the data.

        Parameters
        ----------
        data: (pandas.DataFrame)
            The data to transform.
        """

    def load(self, data, data_file_path):
        """
        A wrapper for pandas.DataFrame.to_parquet

        Parameters
        ----------
        data: (pandas.DataFrame)
            The data to load.
        data_file_path: (str)
            The path to load the dataframe.
        """
        data.to_parquet(data_file_path)

    def run(self, **kwargs):
        """
        Default method to run the task.
        Subclasses may override this method.

        Parameters
        ----------
        **kwargs: Any
            Optional keyword arguments.
            Supported options:
                iterables (list of iterables): A list of iterables to compute
                the Cartesian product. Each tuple in the product represents a
                unique combination of values to iterate over.
        """
        iterables = kwargs.get("iterables", [])

        if len(iterables) < 1:
            raise ValueError("No iterables provided for iteration.")

        iter_product = list(product(*iterables))

        for keys in tqdm(iter_product):
            # Extract file path and handle exceptions
            extract_file_path = self.get_extract_file_path(*keys)

            try:
                data = self.extract(
                    extract_file_path,
                    **kwargs.get("extract", {})
                )
            except FileNotFoundError:
                print(f"File not found: {extract_file_path}. Skipping.")
                continue

            # Transform and Load
            transformed_data = self.transform(
                data,
                *keys,
                **kwargs.get("transform", {})
            )
            load_file_path = self.get_load_file_path(*keys)

            if not transformed_data.empty:
                self.load(transformed_data, load_file_path)
            else:
                print(f"No data for {load_file_path}. Skipping.")

    @abstractmethod
    def get_extract_file_path(self):
        """
        Abstract method to retrieve files or data items to process.
        Subclasses must implement this method with their specific arguments.
        """

    @abstractmethod
    def get_load_file_path(self):
        """
        Abstract method to write path to load the transformed data.
        Subclasses must implement this method with their specific arguments.
        """
