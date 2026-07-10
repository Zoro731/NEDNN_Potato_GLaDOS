# Installation

## Git repository for your group

Before starting, you should have found a group for your project.
Each group should have a single git repository in which you can collaborate for the duration of this course.
At the end of the course, you can hand in all the code you have written by sharing this repository with us.

Please do not clone this repository directly. Instead, each group should decide on one group member that will fork the repository and then add the other group members as collaborators. When forking the repository please name it `MEG-Encoding-{group-name}`, replacing group-name with the name of your group.

If you need help with git, see [this](https://think-like-a-git.net) excellent tutorial, and ask us for help if anything is still unclear.

## Conda environment

0. make sure you have conda installed. You can get anaconda from [here](https://www.anaconda.com/products/distribution). Or alternatively for a smaller installation, you can get miniconda from [here](https://docs.conda.io/en/latest/miniconda.html).

1. clone the course repository.

2. In this repository there is an installation script: `installation.sh`. If you are on Linux or Mac, you can try running it directly (you should navigate to the Code directory and then run the file with `bash install.sh`). If you are on windows or this does not work for some other reason you can run the steps in the script manually. The first step creates a conda environment with some dependencies that are only available through conda. The second command installs the `megcourselib` package for this course (-e specifies that this is an 'editable' package. Any changes to code in `./megcourselib` will be available when importing the package.). The package should install all dependencies it needs automatically (check setup.py to see what they are).

## Setup on the HPC

First, make sure you have sent us your username on the HPC so we can grant you access.

When you first log into the HPC, you may want to familiarise yourself with the system and set up your environment, including conda.
For explanations of the general setup and answers to some frequent questions see [our FAQ](https://github.com/KietzmannLab/SOS).
Do not hesitate to ask us if you are still stuck after reading this document (and if you think others may have the same question, consider asking it in the 'Issues' board of this repository [here](https://github.com/KietzmannLab/MEG-Encoding-Course/issues)).
You can find additional tips in the Documentation folder of this repository (and we may update this folder with answers to questions as they come up).

Once you have access and conda is available, you should be able to follow the same setup instructions as above to set up a copy of the repository and environment on the HPC.
It is most likely easier to complete the setup locally first, make sure everything works, and then repeat the installation on the HPC.
Do NOT install the repository in your home directory on the HPC.
Instead, navigate to the root directory for this course: `cd /share/klab/danthes/MEG_Encoding_Course/`, and create a new directory for your user name: `mkdir $USER`.
Please only work in your own directory. Note that the HPC is a shared resource, and we need to keep it clean and organized.

## Datasets on the HPC

The datasets are already stored on the HPC. Since the raw data is rather large please do not copy whole dataset folders to your project directory.
Instead, Linux allows you to 'symlink' folders. Symbolic links are essentially links to data elsewhere in the file tree that behave like regular files.
They can be created using the `ln -s` command. In the root folder of this repository are two bash scripts that do this for the provided datasets.
Please do not write additional data to these folders as it will change the raw dataset for everyone! (and in fact this should be impossible, since you should not have 'write' permissions in these folders) Instead, if you create your own datasets they should get their own folders in the data directory of your repository.

## Troubleshooting

If you get stuck at any point and the linked resources do not help, please ask us for help. If you encounter an issue with the materials for this course, or think your question may be relevant to others you can also open an issue on this (the MEG-Encoding-Course) repository [here](https://github.com/KietzmannLab/MEG-Encoding-Course/issues). Note though that this repository is public, so do not post any sensitive information here.

Many questions have already been answered in the [FAQ](https://github.com/KietzmannLab/SOS).

For a quickstart specific to this course, see the following links:

- [setting up conda](/docs/TLDR_guides/conda.md)

## Git on the HPC

The HPC does not allow using the (now default) ssh protocol for git. Instead, you will need to use the https protocol.
Since GitHub retired password authentication for https, you will need to use a personal access token.
See [here](https://github.com/KietzmannLab/SOS/blob/main/SOS/GITHUB.md) for instructions on how to do this.
