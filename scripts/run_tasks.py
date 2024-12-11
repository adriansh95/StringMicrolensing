import argparse
import yaml
from utils.task_coordinator import TaskCoordinator
from utils.tasks.kde_label_task import KDELabelTask
from utils.tasks.summary_table_task import SummaryTableTask
from utils.tasks.efficiency_task import EfficiencyTask
from utils.tasks.analyze_backgrounds_task import AnalyzeBackgroundsTask 

if __name__ == "__main__":
    coordinator = TaskCoordinator()

    # Register tasks
    coordinator.register_task("KDELabel", KDELabelTask)
    coordinator.register_task("SummaryTable", SummaryTableTask)
    coordinator.register_task("Efficiency", EfficiencyTask)
    coordinator.register_task("AnalyzeBackgrounds", AnalyzeBackgroundsTask)

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run ETL tasks.")
    parser.add_argument(
        "--config", required=True, help="Path to the YAML configuration file"
    )
    args = parser.parse_args()

    # Load the YAML configuration file
    with open(args.config, "r") as file:
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
