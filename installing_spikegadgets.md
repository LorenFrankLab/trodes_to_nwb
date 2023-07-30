# Installing SpikeGadgets

## Background

SpikeGadgets (Trodes) provide command line tools to export data from the .rec files that were created using their system. During the preprocessing stage of the rec-to-NWB conversion process, these executable files will be called.

So it is required that you have these executable files "installed" on your machine, and that your operation system knows where to look for these files.


## Detailed instructions

### If you have a graphical user interface:

That is, if you are doing this with a display (monitor or X11 forwarding) and keyboard/mouse interaction:

1. **Download.** Go to the [Downloads page](https://bitbucket.org/mkarlsso/trodes/downloads/), and download an installer for **v1.8.2** that matches your system. If you are using Linux, for example, click and download `LinuxOfflineInstaller_1-8-2.tar.gz`.

2. **Run installer.** Extract the tar.gz file, and run the graphical installer and follow the guided steps. If you accept the default location suggested by the installer, this will create a folder named `SpikeGadgets` in your home directory.

3. **Add to PATH.** If you accepted the default location suggested by the graphical installer, you can add this line to your  `~/.profile`, `~/.bash_profile` or `~/.bashrc`:

    ```bash
    PATH="${HOME}/SpikeGadgets:$PATH"
    ```

    These files (any one of them that you can find in your home directory) are just text files. So just open them in your favorite text editor, copy-paste that line at the end of the file, and save.


### If you only have a command line interface:

In this case, I assume that you are connecting to a remote server (maybe a Linux machine) through ssh.


#### If you can use X11 forwarding:

If you can use X11 forwarding (`ssh -X` or `ssh -Y`), it might be easiest to do that for once.

1. **Download.** To download the installer directly to your server, `cd` to a convenient path and do:

    ```bash
    wget https://bitbucket.org/mkarlsso/trodes/downloads/LinuxOfflineInstaller_1-8-2.tar.gz
    ```

    Alternatively, you can download the file locally (where you have the GUI) and copy it to the server.

2. **Install.** This is where you need the X forwarding. If you run the installer by `./[installer-file-name]`, a GUI installer will pop up.

3. **Add to PATH.** If you accepted the default location suggested by the graphical installer, you can add this line to your  `~/.profile`, `~/.bash_profile` or `~/.bashrc`:

    ```bash
    PATH="${HOME}/SpikeGadgets:$PATH"
    ```


#### If you really don't have the GUI:

Unfortunately, the official "installer" for v1.8.x requires the GUI. But what we want is just a collection of executable files and their locations; for example, the default setting of the installer is to create a folder `~/SpikeGadgets` that contains all these files.

So you can simply copy this entire folder and place it in a convenient location on your machine. If you need this ad hoc workaround, do the following:

1. **Download.** Click [this Dropbox sharing link](https://www.dropbox.com/s/680unso15on2wy2/SpikeGadgets-installed.tar.gz?dl=0) to download a tar.gz file. The file size is about 80MB. You may need to download this locally (with a graphical internet browser) and transfer to your server. Haven't tried using wget.
    - This links to someone's personal Dropbox account. Although there is no known concerns, please use this link only when you need & do not put the link in any re-runnable script.
    - If you know of someone who installed the same version on a Linux machine, you can also ask for a copy of their folder.

2. **"Install".** Place this file at a convenient location in your remote server. Simply extract the compressed archive to find a folder named `SpikeGadgets`.

    For example, if you placed the tar.gz file in your home directory, you can do:

    ```bash
    cd ~
    tar -xcvf SpikeGadgets-installed.tar.gz
    ```
    
    This will create a directory `~/SpikeGadgets/` that contains all the executable files you need.

3. **Add to path.** Same as in the other cases. If you are going this route, you probably know what you are doing.
