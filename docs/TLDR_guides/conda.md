# Conda on the HPC

## Installation

On the hpc, conda is available through the package manager 'spack'.
You can load the conda module with the command

```bash
spack load miniconda3
```

This will make the `conda` command available to you. As with your local installation, you need to initialise conda once before you can use it. You can do this by running

```bash
conda init
```

Afterwards open a new shell to make sure the changes take effect. To do this simply run

```bash
bash
```

now you should be able to use conda as you are used to.

## Specify where environments are stored

By default, conda tries to install environments where it was installed. Since we load conda with spack, we do not have write permission in that folder.
Instead, we need to tell conda where to store your packages. You can do this by creating a conda configuration file in your home directory.
Conda should always look for this file when it is used and respect any settings you specify there over the defaults.

The file should be located at `~/.condarc`. You can create it by running:

```bash
touch ~/.condarc
```

This will create an empty file if it did not exist.

Add the following lines to this file:

```yaml
envs_dirs:
  - ~/miniconda3/envs
pkgs_dirs:
  - ~/miniconda3/pkgs
```

Do not worry if these folders do not actually exist, conda will create them when you install your first environment or package.

## Making conda available after login

Packages loaded with spack are generally only loaded for the current session.
Since you will likely want to use conda in every session, you can add the command to your `.bashrc` file.

This is a hidden file in your home directory that is executed every time you open a new shell (file names starting with a . are considered hidden in Linux and are by default not shown in file explorers or when listing files with `ls`. 
You can see them with `ls -a`).

Any commands you add to this file will be executed whenever you log in to the hpc or open a new shell by running `bash`.

To load the conda module every time you open a new shell, you can add the following line to your `.bashrc` file:

```bash
spack load miniconda3
eval "$(conda shell.bash hook)"
```

## Updating conda

If you encounter any issues with this conda installation, the reason might be that the conda version that spack loads is very old. To fix this, in your base environment run:

```bash
conda update conda
```

This should update conda to the latest release.
