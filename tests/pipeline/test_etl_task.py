from pathlib import Path
from unittest.mock import MagicMock
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pipeline.etl_task import ETLTask

class DummyTask(ETLTask):

    def get_extract_file_path(self, *keys):
        return (
            self.extract_dir /
            ("input" + "_".join(map(str, keys)) + ".parquet")
        )

    def get_load_file_path(self, *keys):
        return (
            self.load_dir /
            ("output" + "_".join(map(str, keys)) + ".parquet")
        )

    def transform(self, data):
        return data 

@pytest.fixture
def task(tmp_path):
    return DummyTask(tmp_path, tmp_path)

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "x": [1, 2, 3]
    })

@pytest.fixture
def iterables():
    return [[0, 1], ["a", "b"]]

def test_init():
    dummy_task = DummyTask("extract", "load")
    assert dummy_task.extract_dir == "extract"
    assert dummy_task.load_dir == "load"

def test_extract(task, sample_df):
    input_file_path = task.get_extract_file_path()
    sample_df.to_parquet(input_file_path)

    result = task.extract(input_file_path)

    assert_frame_equal(result, sample_df)

def test_load(task, sample_df):
    output_file_path = task.get_load_file_path()

    task.load(sample_df, output_file_path)

    assert Path(output_file_path).exists()

    result = pd.read_parquet(output_file_path)

    assert_frame_equal(result, sample_df)

def test_run_requires_iterables(task):

    with pytest.raises(ValueError):
        task.run()

def test_run_happy_path(task, sample_df, iterables):
    task.extract = MagicMock(return_value=sample_df)
    task.transform = MagicMock(return_value=sample_df)
    task.load = MagicMock()

    task.run(iterables=iterables)

    assert task.extract.call_count == 4
    assert task.transform.call_count == 4
    assert task.load.call_count == 4

def test_run_keys_passed_correctly(tmp_path, task, sample_df, iterables):
    task.get_extract_file_path = MagicMock(
        side_effect=lambda batch, alpha:
            tmp_path / f"input{batch}_{alpha}.parquet"
    )
    task.extract = MagicMock(return_value=sample_df)
    task.transform = MagicMock(return_value=sample_df)
    task.get_load_file_path = MagicMock(
        side_effect=lambda batch, alpha:
            tmp_path / f"output{batch}_{alpha}.parquet"
    )
    task.load = MagicMock()

    task.run(iterables=iterables)

    task.get_extract_file_path.assert_any_call(0, "a")
    task.get_extract_file_path.assert_any_call(1, "a")
    task.get_extract_file_path.assert_any_call(0, "b")
    task.get_extract_file_path.assert_any_call(1, "b")

    task.get_load_file_path.assert_any_call(0, "a")
    task.get_load_file_path.assert_any_call(1, "a")
    task.get_load_file_path.assert_any_call(0, "b")
    task.get_load_file_path.assert_any_call(1, "b")

def test_run_missing_input_file(task, sample_df, iterables):
    task.extract = MagicMock(side_effect=FileNotFoundError)
    task.transform = MagicMock(return_value=sample_df)
    task.load = MagicMock()

    task.run(iterables=iterables)

    task.transform.assert_not_called()
    task.load.assert_not_called()

def test_run_empty_dataframe(task, sample_df, iterables):
    task.extract = MagicMock(return_value=sample_df)
    task.transform = MagicMock(return_value=pd.DataFrame())
    task.load = MagicMock()

    task.run(iterables=iterables)

    task.load.assert_not_called()
