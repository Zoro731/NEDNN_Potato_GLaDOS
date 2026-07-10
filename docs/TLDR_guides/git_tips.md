# Git Tips

## A useful pattern for safely interacting with git

Especially if you are new to git it can be a bit stressful to perform merges and other git operations that modify your local code.
The easiest way to create a backup for yourself is to make a copy of your local directory.
There is however also a relatively simple pattern you can use to decrease the chances of creating a problem for yourself or losing code:

Say for example you want to merge code that lives on another branch on your remote (e.g. work by another group member), but you think that you will have to resolve merge conflicts that you are unsure about. Instead of directly merging the changes into your current branch and then potentially having to revert commits if you make a mistake while resolving the conflicts you can do the following:

1. From the branch you want to eventually merge into, create a new branch: `git checkout -b merge-branch`. You now have a branch that is an exact copy of the branch you are working on. This command automatically switches your repository to the new branch

2. Perform the merge as you usually would: For example:

```
git fetch origin            # get changes in all remote branches locally
git merge the-other-branch  # merge commits into your merge-branch
```

3. Fix any conflicts & test that everything works as intended.

4. If you are happy, merge your 'helper branch' to the branch you wanted to merge to originally. For example if you wanted to merge into main:

```
git checkout main
git merge merge-branch
```

This merge should never cause any conflicts since your merge-branch was an exact copy of main prior to the merge.

If you did make a mistake and want to 'retry' the merge, simply don't merge your 'helper branch' and delete it instead. You have not modified your original branch, so you can go back to step 1 and try again.

5. (optional) after the merge is complete you can delete your merge-branch to clean up your repository.

## Getting updates from 'upstream'

After forking the course repository your fork no longer automatically receives updates from the 'upstream' repository you forked.
To update your repository with commits from the upstream repository you can use the GitHub interface: On the website for your repository you will see the line 'This branch is $n commits behind KietzmannLab/ML4AnimalCognition:main'.
This means that there are commits that have been made to the upstream repository since you created your fork. You can get these new commits through the 'sync fork' button, however be very careful: this button offers the option to discard all of your own commits to reset your fork back to the state of the remote repository.

### Adding the upstream 'remote' on the command line

Alternatively, you can update your repository on the command line by adding the upstream repository as an additional 'remote'.
Remotes are essentially copies of the repository that exist on another machine you can access (GitHub's servers in this case). The `git remote` command lets you manage these remotes:

```
git remote add upstream git@github.com:KietzmannLab/ML4AnimalCognition.git
```

Creates a new remote, linking to the original repository. You can give it any name you want, in this case it is called 'upstream'.
By default, the remote pointing to your own repository is called 'origin'. You should now see them both with:

```
git remote
```

Next, you have to get all updates from 'upstream' locally by running:

```
git fetch upstream
```

This only downloads the changes and does not modify any code in your current branch. Finally, to pull the changes into your current branch:

```
git merge upstream/main
```

(This is the same syntax as for merges between branches of your own repository).
If the changes from upstream affect the same parts of the code you also modified in your own commits, you may have to resolve any merge conflicts and commit the changes.
