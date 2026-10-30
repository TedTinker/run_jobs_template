I've had a few projects have the same format: train lots of agents with lots of different hyperparameters.

I made this package as a convenient way to set this up. Copy it and edit it like this:

Have a docker image .sif file. Enter its name, and information about your computer cluser, in config.json.

Use bash/slurmcraft.py to make classes of arguments. For example, train agents with every possible combination of alpha values in [1, 2, 3] and beta values in [8, 9, 10].

In utils.py, add defaults for all those hyperparameters.

Update agent.py. Its training function should update q, the proportion of training complete.

Run main.py with a terminal command like this:

bash project_file_name/bash/shell.sh "arg_list=(list of argument names from slurmcraft.py)" "comp=computer_cluster_name" "agents=30"

If there are classes arguments like name_1, name_2, name_3, ..., then name_ will run all jobs of that form.

After jobs complete, finish_dicts.py gathers data saved in training.

Progress in training can be seen with progress.ipynb.

If you're impatient, use GUI.py to make jobs present data early.
