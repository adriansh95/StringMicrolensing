from unittest.mock import MagicMock, call, patch
import pytest
from pipeline.run_tasks import register_all_tasks, main
from test_tasks import DummyTask0, DummyTask1

def test_register_all_tasks_bad_path():
    invalid_path = "/path/that/does/not/exist/"

    with pytest.raises(ValueError):
        register_all_tasks(MagicMock(), invalid_path)

def test_register_all_tasks_no_module(tmp_path):

    with pytest.raises(ImportError):
        register_all_tasks(MagicMock(), tmp_path)

def test_register_all_tasks_no_all():
    path = "tests/pipeline/test_tasks_no_all/"

    with pytest.raises(ImportError):
        register_all_tasks(MagicMock(), path)

def test_register_all_tasks():
    path = "tests/pipeline/test_tasks/"
    task_coordinator = MagicMock()
    calls = [call("Dummy0", DummyTask0), call("Dummy1", DummyTask1)]

    register_all_tasks(task_coordinator, path)

    task_coordinator.register_task.assert_has_calls(calls)

@patch("pipeline.run_tasks.register_all_tasks")
@patch("pipeline.run_tasks.TaskCoordinator")
@patch("argparse.ArgumentParser")
def test_main(mock_parser, mock_coordinator, mock_register):
    mock_parser.return_value.parse_args.return_value = MagicMock(
        task_module="tests/pipeline/test_tasks/",
        task_yaml="tests/pipeline/yamls/valid_tasks.yaml"
    )
    expected_names = ["DummyTask0", "DummyTask1"]
    expected_arguments = [{"alpha": 3}, {"batch": 1}]
    expected_run = [True, False]
    expected_concat = [False, True]

    main()
    mock_coordinator.return_value.run_tasks.assert_called_with(
        expected_names,
        expected_arguments,
        expected_run,
        expected_concat
    )

@patch("pipeline.run_tasks.register_all_tasks")
@patch("pipeline.run_tasks.TaskCoordinator")
@patch("argparse.ArgumentParser")
def test_main(mock_parser, mock_coordinator, mock_register):
    mock_parser.return_value.parse_args.return_value = MagicMock(
        task_module="tests/pipeline/test_tasks/",
        task_yaml="tests/pipeline/yamls/empty_tasks.yaml"
    )

    with pytest.raises(ValueError):
        main()
