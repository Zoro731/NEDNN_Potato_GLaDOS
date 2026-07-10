# Monitoring your running jobs

Once your job is running on the hpc, you can open a new shell on the compute node it is running on to monitor resource usage.

## Connecting to the compute node

To connect to the node, you will first need to find the id of your running job. This id is printed when you first submit the job, or you can find it with the squeue command:

```
squeue -u $USER
```

Then, to get a terminal on the node that is running your job, run:

```
srun --jobid $jobid -s --pty bash
```

replacing $jobid with the id of the job you want to 'log in to'.
If you logged in successfully your terminal prompt should now show the compute node of your running job:

For example, on the login node my prompt looks like this:

```
(base) [danthes@mgmt01 ~]$
```

After connecting to klab-2, one of the cpu nodes, it looks like this:

```
(base) [danthes@klab-2 ~]$
```

## Monitoring memory and cpu usage

After connecting to the compute node, you can run `top` to get a basic 'task manager'. This will show all running processes on the node, and what resources they are currently using.
It can be quite hard to find your own jobs in this list if the node is busy running multiple jobs. To filter for only your own jobs, you can use the `grep` utility:

```
top | grep danthes
```

The above line calls top, and then feeds the output to `grep` (The | operator is called the 'pipe' and lets you chain together commands).
Grep is a search tool that returns all lines matching a string or pattern, so taken together the above command means ("show me all running processes and filter for lines containing 'danthes' (my username)").

This general pattern can be used to get a lot of information quickly. For example:

```
top | grep danthes | grep python
```

Will show me only python processes owned by my user.

```
top | grep danthes | grep python | wc -l
```

Will count them. `wc` is the 'word count' utility and `-l` counts lines instead of words. This can be useful to check whether multiprocessing is working correctly as each worker process gets its own line in `top`. Specifically, if you run a pytorch training script with 8 dataloader workers you would expect the output to be 9 (one main process and 8 dataloaders).

## GPU usage

To check that your script is using the GPU properly, you can use the `nvidia-smi` utility. Contrary to `top`, this command does not automatically refresh so to continuously monitor your gpu usage you can call this command periodically:

```
watch -n 1 nvidia-smi
```

This will run `nvidia-smi` once per second and update the output.
If everything is working, you should see one line per gpu you requested (usually just one). the `Memory-Usage` column tells you how much gpu memory is currently allocated, and the `Volatile GPU-Util` column tells you 'how hard the GPU is working' at the moment. Both should be relatively high and not fluctuate too much if your script is running efficiently. If your memory usage is low, try increasing your batch size. If `GPU-Util` fluctuates a lot and only spikes to high usage then it is likely that you are not feeding data to the GPU fast enough and it is waiting for data to come in. In this case try increasing the number of dataloader workers in your script.

Lastly, the `Processes` section at the bottom tells you which processes are actually using the gpu. There should be at least one process, usually specifying `python`. If this tab is empty but your script is running it is not using the gpu at all.
The `PID` column specifies the 'process id' of your process. This number is a unique identifier for each process running on the machine and is also specified by `top`.
