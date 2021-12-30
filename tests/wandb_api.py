"""Contains functions for interfacing with the WandB API. Used in tests."""

import os
from typing import Optional
import pathlib
import wandb
from wandb.apis.public import Run

WANDB_KEY_FILE = os.path.join(str(pathlib.Path.home()), '.netrc')
WANDB_KEY_VAR = 'WANDB_API_KEY'


def is_wandb_connected() -> bool:
    """Returns True if wandb configuration files are present, False otherwise.

    :return: True if wandb configuration files are present, False otherwise
    """
    if not os.path.exists(WANDB_KEY_FILE) and WANDB_KEY_VAR not in os.environ:
        # No API key file.
        return False
    with open(WANDB_KEY_FILE, 'r', encoding='utf-8') as infile:
        if 'api.wandb.ai' not in infile.read():
            # No wandb entry found in key file.
            return False
    return True


def get_last_wandb_project_run(project_name: str) -> Optional[Run]:
    """Returns the most recent wandb Run associated with the project name, or
    None if no such project exists or there is no wandb configuration.

    :param project_name: The name of the wandb project configured in the model
        wandb callback.
    :return: The most recent wandb Run associated with the project name, or None
        if no such project exists or there is no wandb configuration.
    """
    api = wandb.Api()
    projects = api.projects()
    found_project = None
    for project in projects:
        if project.name == project_name:
            found_project = project
    if found_project:
        runs = api.runs(path=f'{found_project.entity}/{found_project.name}')
        return runs[0]
    return None
