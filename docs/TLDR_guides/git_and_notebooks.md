# Avoiding issues with large notebooks and git

It is not recommended to track large files with git. However, this can easily happen if you have run a Jupyter Notebook that has produced a lot of plots as all the plots will be embedded in the .ipynb file.
Additionally, committing notbooks with their output makes for very messy git diffs and hard-to-resolve merge conflicts.

In general, it makes sense to only use Jupyter Notebooks for exploratory analyses and simple plotting scripts/ data visualization. Once you have a pipelinje you are happy with, export the notebook as a regular python file (`File -> Save and Export Notebook As -> Executable Script`). This has the added benefit that you can use the script in a SLURM job on the cluster.

For notebooks that you do wish to commit to git as part of your final code submission or to share it with your team, I recommend clearing all outputs from the notebook before committing. You can do this in the jupyter interface directly, or with tools like [nbstripout](https://github.com/kynan/nbstripout).
