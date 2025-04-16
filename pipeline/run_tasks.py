"""
run_tasks.py

This script executes a data processing pipeline by registering and running a sequence of tasks.
It is designed to work with a general-purpose pipeline framework defined in `pipeline/` and
task modules defined externally (e.g., in a `tasks/` directory).

Usage:
    python pipeline.run_tasks --task-yaml path/to/task-yaml.yaml --task-module path/to/tasks

Arguments:
    --task-yaml    
        Path to a YAML configuration file specifying the list of tasks to run,
        the order in which to run them, and any arguments to pass to each task.

    --task-module  
        Path to a Python module directory that exposes available task classes
        via its `__init__.py` and `__all__`. The script imports this module,
        extracts the task classes, and registers them with the TaskCoordinator.

Functionality:
    1. Imports the task module specified via `--task-module` and 
       extracts task classes listed in `__all__`.
    2. Registers those task classes with the TaskCoordinator.
    3. Parses the YAML configuration file specified via `--task-yaml`.
    4. Passes task execution instructions to the TaskCoordinator, which 
       manages task instantiation and execution.

Notes:
    - The task module directory must contain an `__init__.py` that 
      defines `__all__` with all task class names.
    - This script is designed to be reusable across projects by simply 
      pointing it to different task-yaml files and task directories.
"""

import os
import sys
import argparse
import importlib
import importlib.util
import yaml
#import utils.tasks
from pipeline.task_coordinator import TaskCoordinator

def register_all_tasks(task_coordinator, tasks_path):
    """
    Dynamically imports and registers all tasks listed in 
    `__all__` from tasks_path
    """
    tasks_abspath = os.path.abspath(tasks_path)

    if not os.path.isdir(tasks_abspath):
        raise ValueError(f"{tasks_abspath} is not a valid directory")

    # Add the parent dir to sys.path so imports work
    sys.path.insert(0, os.path.dirname(tasks_abspath))
    # Infer module name from directory name
    module_name = os.path.basename(tasks_abspath)
    spec = importlib.util.find_spec(module_name)

    if spec is None:
        raise ImportError(f"Can't find module {module_name} in {tasks_abspath}")

    module = importlib.import_module(module_name)

    if hasattr(module, '__all__'):
        for task_name in module.__all__:
            # Dynamically import the task class from module
            task_class = getattr(module, task_name)

            # Register the task with the coordinator
            task_coordinator.register_task(task_name.replace("Task", ""), task_class)
    else:
        raise ImportError(f"Module {module_name} has no __all__ defined.")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run ETL tasks.")
    parser.add_argument(
        "--task-yaml",
        required=True,
        help="Path to the YAML specifying which tasks to run"
    )
    parser.add_argument(
        "--task-module",
        required=True,
        help="Path to task definitions, e.g. tasks or new_project/tasks"
    )
    args = parser.parse_args()
    coordinator = TaskCoordinator()
    # Register tasks
    register_all_tasks(coordinator, args.task_module)

    # Load the YAML configuration file
    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Extract tasks from the configuration
    tasks = config.get("tasks", [])

    if not tasks:
        raise ValueError("No tasks specified in the configuration file.")

    task_names = []
    task_arguments = []
    do_run = []
    do_concat = []

    for task_config in tasks:
        task_names.append(task_config["name"])
        task_arguments.append(task_config.get("arguments", {}))
        do_run.append(task_config.get("do_run", True))
        do_concat.append(task_config.get("do_concat", False))

    # Run tasks
    coordinator.run_tasks(task_names, task_arguments, do_run, do_concat)
