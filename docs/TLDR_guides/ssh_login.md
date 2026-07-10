# SSH

The protocol we use to login to the HPC is called SSH (Secure Shell). 
On Linux and MacOS, the command is usually available by default from your command line.
In recent versions of Windows ssh should also be available from the command line (but I do not have a Windows machine to test this on, please contact us if this does not work!).

At the start of the project you should have shared your uos user name with us so that we can grant you access to the HPC.
You can use your university login credentials to login to the HPC. like this:

```bash
ssh $uos_username@hpc3.rz.uos.de
```

This will prompt you for a password, and then log you in to the HPC.

## Log in without a password

It can get quite tedious to enter your password every time you want to log in to the HPC.
Instead, you can set up a key pair that will allow you to log in without a password:

For this you need to generate a key pair on your local machine. SSH provides a set of tools for this:

```bash
ssh-keygen 
```

To create a new public/private key pair. Optionally you can set a passphrase for your key pair, or specify where to store it 
(on Unix systems these usually live in your home directory in: `~/.ssh/`).

This will create two files: `id_rsa` (your private key) and `id_rsa.pub` (your public key). (If you changed the name of the key, the files will be named accordingly).

Add the key to your ssh keyring:

```bash
ssh-add ~/.ssh/id_rsa
```

Now you need to give the **public** key to the HPC. You can do this with a single command:

```bash
ssh-copy-id -i ~/.ssh/mykey $uos_username@hpc3.rz.uos.de
```

Make sure to never share your private key, as this would allow anyone to log in to your account.
The command will ask you for your password once more, and then copy the public key to the HPC.

Now you should be able to log in without a password with the same ssh command as before.

## SSH config file

To make logging in even easier, you can add a configuration entry to your ssh config file.
By default this should be in `~/.ssh/config` (if it does not exist, you can create it).

```bash
Host osna-hpc
HostName hpc3.rz.uos.de
User=$uos_username
IdentityFile = /path/to/your/private/key
ForwardAgent yes
AddKeysToAgent yes
```

You can now log in by specifying the value in Host:

```bash
ssh osna-hpc
```

and ssh should automatically use the correct username and key.

This alias will also work in commands such as `scp` and `rsync` (explained [here](copying_files.md)):

```bash
rsync -av /path/to/local/file osna-hpc:/path/to/remote/directory
```


