"""
Task Coordinator Module

This module provides a `TaskCoordinator` class that facilitates the execution 
of ETL tasks. It supports dynamic task registration, configuration, and 
execution based on user-defined parameters. Tasks can optionally be run 
and/or have their results concatenated.
"""

class TaskCoordinator:
    """
    A coordinator for executing ETL tasks in a controlled and flexible manner.

    The `TaskCoordinator` allows dynamic registration of ETL tasks and provides 
    a mechanism to execute tasks with specified configurations. It supports 
    initializing tasks with arguments, running their `run` methods, and 
    optionally calling their `concat_results` methods.
    """
    def __init__(self):
        self.registry = {}

    def register_task(self, name, task_cls):
        """
        Registers a task class with the coordinator.

        Args:
            name (str): A unique name for the task.
            task_cls (type): The task class to register. Must be a subclass 
                             of `ETLTask`.

        Raises:
            ValueError: If a task with the given name is already registered.

        Example:
            coordinator.register_task("my_task", MyTaskClass)
        """
        self.registry[name] = task_cls

    def run_tasks(self, task_list, task_arguments, task_run, task_concat):
        """
        Executes a sequence of registered tasks with specified configurations.

        Args:
            task_list (list[str]): A list of task names to execute. Each name 
                                   must correspond to a registered task.
            task_arguments (list[dict]): A list of dictionaries containing 
                                          arguments for each task. Each 
                                          dictionary should have the following keys:
                                          - "init_args" (list): Positional arguments 
                                            for initializing the task.
                                          - "init_kwargs" (dict, optional): Keyword 
                                            arguments for initializing the task.
                                          - "run_kwargs" (dict, optional): Keyword 
                                            arguments for the task's `run` method.
            task_run (list[bool]): A list of booleans indicating whether the 
                                   `run` method should be called for each task.
            task_concat (list[bool]): A list of booleans indicating whether the 
                                      `concat_results` method should be called 
                                      for each task.

        Raises:
            ValueError: If a task in `task_list` is not found in the registry.

        Example:
            task_list = ["task1", "task2"]
            task_arguments = [
                {"init_args": ["/input1", "/output1"], "run_kwargs": {"batch_range": (0, 10)}},
                {"init_args": ["/input2", "/output2"], "init_kwargs": {"option": True}}
            ]
            task_run = [True, True]
            task_concat = [False, True]
            coordinator.run_tasks(task_list, task_arguments, task_run, task_concat)
        """
        for task_name, task_args, do_run, do_concat in zip(
            task_list,
            task_arguments,
            task_run,
            task_concat
        ):
            init_args = task_args["init_args"]
            init_kwargs = task_args.get("init_kwargs", {})
            run_kwargs = task_args.get("run_kwargs", {})

            if task_name not in self.registry:
                raise ValueError(f"Task '{task_name}' not found in registry.")

            task_cls = self.registry[task_name]
            task_instance = task_cls(
                *init_args,
                **init_kwargs
                )

            if do_run:
                task_instance.run(**run_kwargs)

            if do_concat:
                task_instance.concat_results()
