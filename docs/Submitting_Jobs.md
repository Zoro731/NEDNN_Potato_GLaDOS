# Submitting Jobs on the HPC

The HPC is set up with [Slurm](https://slurm.schedmd.com), a scheduling software to fairly manage resource allocation.

When you first log in to the HPC your terminal shows a session running on the login node. You can see this in your shell prompt. It should look something like this:

```
[danthes@mgmt01 ~]$
```

The name of the login node is `mgmt01`. This node is not meant to run any large scripts, and resource use is very limited for each user.
To run your scripts, you submit them to the scheduler, along with information on which resources you request, and for how long. It is good to provide a realistic estimate of the resources you will actually need: If you request too little resources, or too little time, your job will be cancelled by the scheduler. If you request too many resources, it can take longer for your job to run since you will have to wait until all the resources you request are available. Unfortunately, there is often not an objective way to know exactly how many resources your job will need, but try to estimate what you will actually need. Once you run the job for the first time you can get a better idea of the required memory (one way to know for sure is to request a conservative amount of memory, and then increase the requested resources until the job no longer crashes, however this is impractical if you expect the most resource intense parts of your script will be towards the end of a long running job).

There are a number of submission scripts in `./submit/` that you can adapt for your purposes. Good starting points are `submit_to_cpu.sh` and `submit_to_gpu.sh`.
You will learn how to submit a job using these scripts below - make sure to completely read these instructions before you continue!

## Example

Slurm submission scripts always start with a line indicating that the script is a bash script (bash is the 'language of your terminal' when you log in to the HPC).
The next lines give information to the Slurm schedule, and always start with #SBATCH. A few key parameters are:

- `-p` specifies the 'partition' you submit to. Our compute nodes are organized in partitions based on the resources they have. The partition `workq` is the university queue for jobs needing a cpu. In the gpu job example you will see `klab-gpu` instead, which is the partition containing the lab's compute nodes with gpus (do not submit scripts that only need a cpu to the gpu queue as you will take away resources from jobs that need a gpu).

- `-t` allows you to specify the time limit for your job in the format `d-hh:mm:ss`. Try to be conservative here so that your jobs run quickly, but don't underestimate the runtime of your script because it will get cancelled after this timeout.

- `--mem` allows you to specify the amount of RAM you need

- `-c` specifies the number of cpu cores. If you use multiple, make sure that your script actually can make use of parallelisation. Otherwise the remainder of the cores will sit idle

An example 'header' for a slurm submission script. This example will request 8 cpu cores and 64gb of memory for 30 minutes.

```
#!/bin/bash
#SBATCH -p workq
#SBATCH -t 0-00:30:00
#SBATCH --mem=64G
#SBATCH -c 8
```

The next few lines set up your environment.
Since your permissions on the HPC are restricted you cannot (and should not attempt to!) install additional software outside your python environment. However, many necessary programs are available through the spack package manager. The below example loads git and cuda dependencies (needed for gpu jobs only).

```
spack load git
spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8
```

After all the setup is done, you can execute any commands as you would in your terminal.
The command below is a print statement, followed by running the 'hello-world' script you ran on your laptop as well when you set up the repository locally.

```
# this can be any command, use like your cmdline

echo "Setup complete, running hello-world script ..."
uv run scripts/hello-world.py
```

The full example submission script is included in this repository here: [submit/submit_to_cpu_example.sh](/submit/submit_to_cpu_example.sh). In the next section, we will learn how to submit it to the queue

## Submitting your job

To submit your job to the scheduler you use the `sbatch` command, pointing to the submission script you just wrote.
As with all the code included with this repository these scripts are intended to be run from the root directory of this repository.
For example, to submit this example script you would run:

```
sbatch submit/submit_to_cpu_example.sh
```

The folder also includes an analog example script for a job that requires a GPU. For this script, please do not change the `--gres` line specifying the kind of GPU the script requests. If you want to extract features from a large network, please consult us and we can see whether we can make a larger GPU available.

## Monitoring your jobs

Since you are not running your script directly, you also will not see the output in your current terminal.
Instead, all the output will be written to a file in the folder from which you submitted the job.

First, make sure your job is running:

```
squeue -u $USER
```

This will show you all the jobs you submitted to the queue.
Jobs that are running will have the compute node and time since they started listed. If your job is listed with (PRIORITY) or (RESOURCES) that means it is still queued and has not started.

You can also run

```
squeue -p klab-gpu
```

To see jobs by all users that are queued for a partition (change to `workq` to see the cpu queue instead).

Every job also has a jobid which you can use to cancel the job:

```
scancel $jobid
```

You can also cancel all of your jobs with

```
scancel -u $USER
```

This ID will also allow you to find the output the job is producing. Any output will be stored in files called

```
slurm-$jobid.out
```

With `$jobid` replaced by the ID you found with `squeue`.

To print all output generated so far to your terminal, you can use:

```
cat slurm-$jobid.out
```

If you want to monitor this file as new output is generated while your job is running, you can use:

```
tail -f slurm-$jobid.out
```

`tail` prints the last few lines of long text files with the `-f` option telling the command to 'follow' as content is appended to the file.
