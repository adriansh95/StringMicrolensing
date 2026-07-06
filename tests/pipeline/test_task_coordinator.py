from unittest.mock import MagicMock
import pytest
from pipeline.task_coordinator import TaskCoordinator

task_instances = []

class DummyTask():
    task_instances = []

    def __init__(self, *args, **kwargs):
        self.init_args = list(args)
        self.init_kwargs = kwargs
        self.run = MagicMock()
        self.concat_results = MagicMock()

        DummyTask.task_instances.append(self)
        
@pytest.fixture
def task_coordinator():
    return TaskCoordinator()

def test_init(task_coordinator):
    assert task_coordinator.registry == {}

def test_register_task(task_coordinator):
    task_coordinator.register_task("DummyTask", MagicMock)
    assert task_coordinator.registry["DummyTask"] == MagicMock 

def test_run_tasks(task_coordinator):
    DummyTask.task_instances.clear()

    task_coordinator.register_task("DummyTask0", DummyTask)
    task_coordinator.register_task("DummyTask1", DummyTask)

    task_list = ["DummyTask0", "DummyTask1"]
    task_arguments = [
        {
            "init_args": ["dummy0_init_arg0", "dummy0_init_arg1"],
            "run_kwargs": {"dummy0_kwarg0": "alpha"}
        },
        {
            "init_args": [
                "dummy1_init_arg0", "dummy1_init_arg1", "dummy1_init_arg2"
            ],
            "init_kwargs": {"option": False},
            "run_kwargs": {"dummy1_kwarg0": "beta", "dummy1_kwarg1": "delta"}
        }
    ]
    task_run = [True, True]
    task_concat = [False, True]

    task_coordinator.run_tasks(
        task_list,
        task_arguments,
        task_run,
        task_concat
    )

    dummy_task_0, dummy_task_1 = DummyTask.task_instances

    assert dummy_task_0.init_args == task_arguments[0]["init_args"]
    assert dummy_task_0.init_kwargs == {}
    assert dummy_task_1.init_args == task_arguments[1]["init_args"]
    assert dummy_task_1.init_kwargs == task_arguments[1]["init_kwargs"]

    dummy_task_0.run.assert_called_with(**task_arguments[0]["run_kwargs"])
    dummy_task_1.run.assert_called_with(**task_arguments[1]["run_kwargs"])

    dummy_task_0.concat_results.assert_not_called()
    dummy_task_1.concat_results.assert_called_once()

def test_run_tasks_raises_valueerror(task_coordinator):
    DummyTask.task_instances.clear()

    task_list = ["DummyTask2"]
    task_arguments = [{"init_args": [], "run_kwargs": {}}]
    task_run = [True]
    task_concat = [False]

    with pytest.raises(ValueError):
        task_coordinator.run_tasks(task_list, task_arguments, task_run, task_concat)
