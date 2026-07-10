# Editing Files on the HPC

You usually connect to the HPC via SSH, which means that there is no graphical interface to edit files.
When working on your project the easiest solution is to just work on your own computer in your favourite editor, testing all your code runs as you expect, and then synchronising it with the HPC with git (this has the nice side effect of keeping a history of your changes and making sure you don't lose your changes!).

However, for configuring your environment or making quick changes to configuration files this is not practical.
Instead, you can use one of the installed text editors that work in the terminal. 
The one installed by default on the HPC is `vim`.

For an extensive primer on vim, see [here](https://missing.csail.mit.edu/2020/editors/).

Or if you really just want to make a quick change:

1. open vim by typing `vim` in the terminal
    - typing vim followed by a filename will open that file in vim, e.g. `vim my_file.txt`, this can be a path, too, e.g. `vim /path/to/my_file.txt`

2. vim works with modes. By default you are in 'normal mode' which allows you to move around a file and do some basic operations. 
Move the cursor with your arrow keys or `h`, `j`, `k`, `l` (left, down, up, right).
To start editing, you need to switch to 'insert mode'. You can do this by pressing `i`. 
You will see your cursor change, and the bottom of your screen will show `-- INSERT --`.
Now you can type as you would in any other editor.

3. When you are done editing, press `ESC` to go back to 'normal mode'.

4. In normal mode you can send commands to vim by typing `:` followed by a command.
    - to save your changes type `:w`
    - to exit vim type `:q`
    - to save and exit in one go type `:wq`

5. If you are stuck, you can always exit vim by typing `:q!` to discard your changes and exit (you will lose all your edits).

## Bonus commands

All commands are executed in normal mode:

- `dd` deletes the current line
- `yy` copies the current line
- `p` pastes the copied or deleted line below the current line
- `P` pastes the copied or deleted line above the current line
- `u` undoes the last change
- `gg` moves the cursor to the beginning of the file
- `G` moves the cursor to the end of the file
- `/<search term>` searches for the next occurrence of `<search term>`. Press `n` to find the next occurrence.
- `A` moves the cursor to the end of the line and switches to insert mode
