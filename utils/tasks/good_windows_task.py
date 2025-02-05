"""
This module defines GoodWindowsTask. GoodWindowsTask uses
utils.lc_scanner.LcScanner to scan lightcurves and record
all the time intervals during which a lensing event could
start and lens a sufficient subset of the samples given
the achromaticity requirements.
"""
import os
from utils.tasks.etl_task import ETLTask

class GoodWindowsTask(ETLTask):
    """
    Template task defining methods.

    Attributes:
        extract_dir (str): Directory containing the input data files to be 
                           processed.
        load_dir (str): Directory where the transformed data files will be 
                        written.

    Methods:
        get_extract_file_path(*keys):
            Returns the file_path to process given the keys.
        get_load_file_path(*keys):
            Returns the file_path to load given the keys.
        transform(data, *keys):
            Transform the data.
    """
    def transform(self, data, i_batch):
        """
        Transform the data.

        Parameters:
        ----------
        data: `pandas.DataFrame`
            The data to transform.
        i_batch: `int`
        """

    def get_extract_file_path(self):
        """
        Get the extract file path.

        Parameters:
        ----------
        """
        result = os.path.join(
            self.extract_dir,
            "extract_stem.parquet"
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
        """

    def get_load_file_path(self):
        """
        Get the load file path.

        Parameters:
        ----------
        """
        result = os.path.join(
            self.extract_dir,
            "load_stem.parquet"
        )
        return result

