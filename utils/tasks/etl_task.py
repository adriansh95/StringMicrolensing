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
    """
    def __init__(self, extract_dir, load_dir):
        self.extract_dir = extract_dir
        self.load_dir = load_dir

    def extract(self, data_file_path, **kwargs):
        """Read in the parquet file"""
        result = pd.read_parquet(data_file_path, **kwargs)
        return result

    @abstractmethod
    def transform(self, data):
        """
        Abstract method to transform the data to data_file_path.
        """

    def load(self, data, data_file_path):
        """Write the transformed data to the load_dir."""
        data.to_parquet(data_file_path)

    def run(self, **kwargs):
        """
        Default method to run the task.
        Subclasses may override this method.
        """
        iterables = kwargs.get("iterables", [])

        if len(iterables) < 1:
            raise ValueError("No iterables provided for iteration.")

        iter_product = list(product(*iterables))

        for keys in tqdm(iter_product):
            # Extract file path and handle exceptions
            extract_file_path = self.get_extract_file_path(*keys)

            try:
                data = self.extract(extract_file_path, **kwargs["extract"])
            except FileNotFoundError:
                print(f"File not found: {extract_file_path}. Skipping.")
                continue

            # Transform and Load
            transformed_data = self.transform(data, *keys)
            load_file_path = self.get_load_file_path(*keys)
            self.load(transformed_data, load_file_path)

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
