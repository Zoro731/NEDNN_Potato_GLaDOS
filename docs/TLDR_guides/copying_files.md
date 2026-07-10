# Copying files to and from the HPC

When working with the HPC you may need to copy files between your local machine and the HPC.
For all of your code, it makes sense to rely on git since this will keep a history of your changes and make sure you don't lose your work.
It also facilitates collaboration with your group members.

However, do not try to commit data or other larger files to git. Githut will refuse to accept large files (~500MB) and it is generally not a good idea to store large files in git.
Instead, there are two main ways to conveniently synchronise files between your local machine and the HPC:

## Command Line

### scp

`scp` is a command line tool that allows you to copy files between two machines. It should be pre-installed on most Linux and MacOS systems.
To copy a file from your local machine to the HPC, you can use the following command:

```bash
scp /path/to/local/file $remote_address:/path/to/remote/directory
```

$remote_address is the address of the HPC, it will be the same address you use to connect via SSH.
(If you have [configured logging in without a password](ssh_login.md)), you can use the same shortcut address here.


The same principle holds in the other direction. To copy a file back to your local machine:

```bash
scp $remote_address:/path/to/remote/file /path/to/local/directory
```

### rsync

`rsync` is another command line tool that allows you to synchronise files between two machines.
It's advantage over scp is that it only copies files that have changed, which can save a lot of time if you are working with large files or many files.
It's also more robust to broken connections and will often resume copying where it left off.

To copy a file from your local machine to the HPC, you can use the following command:

```bash
rsync -av /path/to/local/file $remote_address:/path/to/remote/directory
```

Again, to go in the other direction:

```bash
rsync -av $remote_address:/path/to/remote/file /path/to/local/directory
```

rsync may not be installed on your local machine by default, but it should be available for all Unix devices.

*BE CAREFUL:* any copying done frome the command line will not ask you about overwriting files. It will simply overwrite them. Make sure you are copying the correct files and that you are not overwriting anything important.

## Graphical Interface

Sometimes it is more convenient to use a graphical interface to copy files.
A good tool for this is [Cyberduck](https://cyberduck.io/), which is available for both MacOS and Windows.

