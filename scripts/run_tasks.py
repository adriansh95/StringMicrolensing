import argparse
import yaml
import utils.tasks
from utils.task_coordinator import TaskCoordinator

def register_all_tasks(task_coordinator):
    """
    Dynamically imports and registers all tasks listed in `__all__` from utils.tasks.
    """
    for task_name in utils.tasks.__all__:
        # Dynamically import the task class from utils.tasks
        task_class = getattr(utils.tasks, task_name)

        # Register the task with the coordinator
        task_coordinator.register_task(task_name.replace("Task", ""), task_class)

if __name__ == "__main__":
    coordinator = TaskCoordinator()

    # Register tasks
    register_all_tasks(coordinator)

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run ETL tasks.")
    parser.add_argument(
        "--config", required=True, help="Path to the YAML configuration file"
    )
    args = parser.parse_args()

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
