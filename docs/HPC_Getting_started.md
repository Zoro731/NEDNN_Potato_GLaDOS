# HPC Setup

For the second challenge, you can use the HPC to compute more expensive models.
See the instructions below for instructions to get you started.

## Step 0: Disclaimer

With great power comes great responsibility! The HPC is a shared resource - please make sure to be 'good citizens' and read the guide below carefully before you continue. Additionally, please mind your resource usage, especially regarding storage: As you likely have already noticed, running feature extraction for deepnets on large datasets can create very large files. Make sure you share and re-use these large datasets where applicable and try not to create duplicates within your group. You can find tips for this below, but always feel free to ask us if you are unsure about anything you want to do on the HPC!

### More Resources

There are additional resources available that already have answers to common questions. Please take a look at them:

- 'SOS', the lab internal HPC documentation: https://github.com/KietzmannLab/SOS
- [Guides in this repository](/docs/TLDR_guides)

## Step 1: Logging in

If you have shared your uos username with us you should have been granted access to the HPC. You can log in using `ssh` if you are connected to the university network. If you would like to access the HPC from outside the university network, you will need to activate the `eduvpn` program first.

To log in, in a terminal run the following command (replacing $youruser with your uos username). The password will be the same as for other university services.

```
ssh $youruser@hpc3.rz.uni-osnabrueck.de
```

## Step 2: Navigate to the correct repository

After logging in, you will be directed to your 'home' directory. Importantly, you should not store any data there! Instead, we have created a folder for this course. You can navigate to this folder like this:

```
cd /share/klab/danthes/MEG_Encoding-sose2026/projects/
```

This folder has prepared folders Group{1-5}. You should have write access to these folders and all your files should live in your Group's directory. Feel free to create a 'personal' folder if you need to keep your code separated among group members.

Once you have settled on a folder structure that works for your group, you can clone your project git repository to access your code.

## Step 3: Getting the data

The data for the second challenge is available in the folder `/share/klab/danthes/MEG_Encoding-sose2026/shared_data`. This folder is read-only to you, so the data can safely be used by all groups. As with the 'local' setup you completed at the beginning of the course, the data needs to be available in the `data` folder in your repository. To avoid copying the data repeatedly, you can 'link' it instead (this is the analog to creating a shortcut to a program or folder on your laptop). There is a file called `HPC_SETUP.sh` in the root of this repository that will create the links for you. Run it as `bash HPC_SETUP.sh` from the root directory of your repository.
If you get a permission error, the file needs to be marked executable first. You can do this like this: `chmod +x HPC_SETUP.sh`

## Step 4: Installing uv

You can install uv exactly as you did on your own machine by running:

`curl -LsSf https://astral.sh/uv/install.sh | sh`

## Step 4: Submitting Jobs

Now your setup should be complete! Next, you can learn how to submit your first job.
Instructions for submitting scripts using the `SLURM` scheduler used on the HPC are [here](/docs/Submitting_Jobs.md). Carefully read this document and afterwards try submitting the example script for a cpu job.
