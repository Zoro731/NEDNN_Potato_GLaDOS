# Setup

## Fork or clone this repository

This repository is designed to be a template for your projects. To ensure that every group has their own repository to work with, please do not clone this repository directly.

Instead, you should create a 'fork'. You can do this directly on GitHub:

![forking a repo](/docs/imgs/fork.png)

This will create a new copy of the repository in your own GitHub account.
To collaborate with your group, one group member should fork the repository and add the other group members as collaborators.

## Cloning the repo

To continue, you need a local copy of this repository on your computer. If you do not know how to 'clone' a repository, please ask us or your class mates for help. For more detailed instructions on how Git works, links to our favorite tutorials can be found (here)[/docs/HELP.md].

## Before you continue

For the next steps, we assume you have a local copy of your repository opened in a terminal.
You can open a terminal on your laptop and navigate to the folder directly.
Alternatively, especially if you have never done this before you can open the repository folder in your editor of choice (e.g. VS Code). Most editors also allow you to open a terminal in the project folder directly.

## Set up your python environment

### Installing the environment

This repository contains a `pyproject.toml` file that lists all Python dependencies (you may add additional dependencies as you work on your projects).
To set up the project, we have used the `uv` package manager. You can follow the steps below to create an exact copy of our Python environment, first install uv if it is not already installed on your machine. (Installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/)).
You can then directly run python commands or a Jupyter notebook server.

#### running a python script

`uv run scripts/hello-world.py`

The first time you run this command, all packages should be installed automatically. Then, this should print a welcome message and no errors. This means your environment has been created.

#### starting a jupyter notebook server

`uv run --with jupyter jupyter lab`

This should open the jupyter lab environment in your browser. If it does not, a link should be displayed in your terminal (along the lines of localhost:8888). Click it, and the notebook environment should open. Then, navigate to the notebooks folder and open the notebook hello-world.ipynb and run the first cell in that notebook. If you do not get an error here either, you are good to go!

#### INstalling additional packages

If you need additional packages, you can install them with `uv add <package-name>`. The next time you run a script, uv will update your environment automatically.

### Additional information on uv

If you are interested why this works, there is more information on the paragraph below - this can safely be skipped for now:

The repository contains a `pyproject.toml` file. When you run commands with uv, the program checks for this file in the current folder. If it finds it, it will check all the dependencies listed, and download them if they are not already installed. The installed packages are included in the folder `.venv` in your copy of the repository. The first time you run a command with uv, an additional `uv.lock` file gets created that lists the exact versions of all the package versions you have installed. This ensures that your environment is reproducible: When you cloned / forked this repository, you also got a copy of my `uv.lock` file. This ensures that your environment is an exact copy of mine. This hopefully prevents any issues with version mismatches. (e.g. in the `pyproject.toml`, we often only specify the minimum version of a package like this: `numpy >= 2.0`. If numpy releases a new version that breaks old functionality your environment might be broken if you try to use this repository in the future).

## Get the data

You should have already gotten a link to download the data for the first challenge. Download it and extract all the zip files to the `./data` folder in you copy of the repository on your own computer. The folder structure should look like this:

```
MEG_Encoding_course
├── data
│   ├── avs_scenes
│   └── brainencoding26
│       ├── challenge1
│       │   ├── subject60
│       │   │   ├── challenge1_dev
│       │   │   ├── challenge1_eval
│       │   │   └── ground_truth
│       │   └── training
...
```
