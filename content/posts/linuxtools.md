+++
authors = ["Mike Harmon"]
title = "Linux Tools For Data Science"
date = "2017-10-05"
tags = [
    "Linux",
    "Terminal",
    "Shell Scripting",
    "Git",
    "Vim"
]
series = ["Linux"]
aliases = ["migrate-from-jekyl"]
+++

## Contents
---------------------

__[1. Introduction](#Introduction)__

__[2. Basic Commands](#Basics)__

- **Topics:** ls (-a, -al), pwd, mv, cd, cp, ps, top, kill, htop, history, man

__[3. Files And Vim](#Files)__

- **Topics:** mkdir, rmdir, touch, rm (-rf), less, grep, vi, chmod

__[4. Working Over A Network](#Network)__

- **Topics:** ssh, tar, sftp, screen 

__[5. A Word On Working With Git](#Git)__

__[6. Conclusion And More Resources](#More)__


## Introduction <a class="anchor" id="Introduction"></a>
-------------------

Knowing Unix tools and Bash commands is not the sexiest part of data science and is often the most overlooked skills.  During my time as a Ph.D. student in Computational Science/Applied Mathematics I picked up a bunch of unix commands that were life savers and I'm going to go over a few here. Learning these skills is definitely can seem a little bit boring, but I cannot emphasize how useful they are. Setting up my unix environment and linking various libraries was one of the most frustrating parts of graduate school, but I believe I am much more productive as a data scientist having learned these valuable lessons.  



Most of the commands and concepts I will be going over don't require any special libraries and when they do I'll provide links to them.  In fact most of the unix commands can be run from <a href="http://jupyter.org/">Jupyter Notebook</a>.  When they don't, I will run them from the <a href="https://en.wikipedia.org/wiki/Terminal_(macOS)">Terminal</a>, which is the MacOS version of the <a href="https://en.wikipedia.org/wiki/Unix_shell">Unix/Linux shell</a>.


## Basic Commands <a class="anchor" id="Basics"></a>
-------------------

One of my favorite resources for learning various scientific computing concepts during graduate school was the <a href="http://www.math.colostate.edu/~bangerth/videos.html">deal.ii video lecture series</a>.  While these videos were made for scientific computing using their library, they still are extremely good resources for data scinentist beause the authors are incredibly thorough and amazing educators.  For videos on command basics see this <a href="http://www.math.colostate.edu/~bangerth/videos.676.2.9.html">link</a>.  

The first command to we'll go over in this post is,

- **<code> ls </code>**


<code>ls</code> lists all the files in the current *directory* (this is just a fancier way of saying "folder"):


```python
ls
```

    [34mDirec1[m[m/           Unix_Tools.ipynb  file1


We see that there is a directory called <code>Direc1/</code> (the forward slash after the name gives away that it is a directory instead of a file) and two files: this notebook (<code>Unix_Tools.ipynb</code>) as well as a file called <code>file1</code>. We can view the "hidden files" (those with a . infront of them) using a **<code>/-a</code>** after the  <code>ls</code>:


```python
ls -a
```

    [34m.[m[m/                  [34m.ipynb_checkpoints[m[m/ Unix_Tools.ipynb
    [34m..[m[m/                 [34mDirec1[m[m/             file1


The </code>.ipynb_checkpoints/</code> is a "hidden directory."  The <code>./</code> stands for the current directory (we could also use <code>ls .</code> instead of <code>ls</code>). And the <code>../</code> stands for the parent directory (the directory containing this directory). 

We can get more information on the files and directories using the <code>ls -al</code> command (this will give us information on all the files, if we wanted just non hidden ones we would use <code>ls -al</code>): 


```python
ls -al
```

    total 24
    drwxr-xr-x   6 Mike  staff   204 Jul 16 20:54 [34m.[m[m/
    drwxr-xr-x  14 Mike  staff   476 Jul 16 14:30 [34m..[m[m/
    drwxr-xr-x   3 Mike  staff   102 Jul 16 14:03 [34m.ipynb_checkpoints[m[m/
    drwxr-xr-x   4 Mike  staff   136 Jul 16 20:45 [34mDirec1[m[m/
    -rw-r--r--   1 Mike  staff  9944 Jul 16 20:52 Unix_Tools.ipynb
    -rw-r--r--   1 Mike  staff     0 Jul 16 20:54 file1


Each row now corresponds to a file or directory. It also gives us information on the permisions for the file/directory, number of links, owner name, group name, number of bytes in the file, abbreviated month, day-of-month file was last modified, hour file last modified, minute file last modified, and the pathname/file name. 

We can also view the path to the current directory using,

- ** <code> pwd </code> **


```python
pwd
```




    u'/Users/Mike/Documents/DS_Projects/Unix_Tools'



You can see that in jupyter notebooks the path is returned as unicode. We can also use <code>ls</code> to view the contents of other directories than our current one.  We can see the contents of <code>Direc1/</code> by typing:


```python
ls Direc1
```

Nothing happened here because that directory is emtpy.  We can move the file, <code>file1</code>, into <code>Direc1/</code> by using the command

- ** <code>mv </code>**


```python
mv file1 Direc1/
```

We can now see the contents of <code>Direc1</code> again to see that the file has moved there:


```python
ls Direc1/
```

    file1


We can then go into to <code>Direc1</code> by using,

- **<code>cd</code>** 

which stands for "change directory,"


```python
cd Direc1/
```

    /Users/Mike/Documents/DS_Projects/Unix_Tools/Direc1


We can also use <code>mv</code> to *change the name of a file or directory*:


```python
mv file1 file2
```


```python
ls 
```

    file2


We can copy the contents of <code>file2</code> into <code>file1</code> using the command,

- **<code>cp</code>**


```python
cp file2 file1
```

We can then go back to the the parent (original) directory using,


```python
cd ..
```

    /Users/Mike/Documents/DS_Projects


We can see the process that are running in this directory,

- **</code>ps</code>**

This one we will have to use in the terminal,

<img src="https://github.com/mdh266/Unix_Tools/blob/master/images/ps.png?raw=1">

The <code>PID</code> is the process id and is important because we can use it to help us <a href="https://en.wikipedia.org/wiki/Kill_(command)">kill</a> the *process* or *command* if we need to. The <code>Time</code> is how long the process has been running and <code>CMD</code> is the name of the *command* or process that is running.  The <code>TTY</code> isn't something that I have ever have had to use.

We can also use the command,

- **<code>top</code>**

to see all the processes running on our computer, the results from my terminal are below,

<img src="https://github.com/mdh266/Unix_Tools/blob/master/images/top.png?raw=1">

As you can see theres a lot more information that is provided by <code>top</code> including the ammount of memory each process is using.  One tool I liked using in graduate school is, 

- **<a href="http://hisham.hm/htop/">htop</a>**

which provides an interactive version of of <code>top</code>.  I liked it because when writing multi-threaded applications you can see directly how much work each core/thread is using (you can get a similar effect using  <code>top</code> by pressing <code>1</code> while <code>top</code> is running).  An example on my computer of the results of <code>htop</code> are shown below,

<img src="https://github.com/mdh266/Unix_Tools/blob/master/images/htop.png?raw=1">



The last two basic commands I'll mention are,

- **<code>history</code>**

which shows use a list of all the commands you have used recently.  As well as,


- **<a href="https://en.wikipedia.org/wiki/Man_page">man</a>**

which can be used to show the manual page of specific unix commands. 

Now that we have on the basics of unix commands we can move on to dealing with directories and files more concretely.

## Files And Vim<a class="anchor" id="Files"></a>
-------------------

### Working with Files:

We can create directories using the,

- **<code>mkdir</code>**

command.  Say we want to create a new directory called <code>Direc2/</code>, we can do this by the command,


```python
mkdir Direc2/
```


```python
ls
```

    [34mDirec1[m[m/           README.md         file1
    [34mDirec2[m[m/           Unix_Tools.ipynb  [34mimages[m[m/


We can remove empty directories by using,

- **<code>rmdir</code>**

as we see below,


```python
rmdir Direc2/
```


```python
ls
```

    [34mDirec1[m[m/           Unix_Tools.ipynb  [34mimages[m[m/
    README.md         file1


Next let's go into directory <code>Direc1/</code> and create a file.


```python
cd Direc1/
```

    /Users/Mike/Documents/DS_Projects/Unix_Tools/Direc1



```python
ls .
```

    file1  file2


We can make an empty by using the command,

- **<code>touch</code>**

Let's make a file called <code>file3.txt</code> using the terminal,

<img src="https://github.com/mdh266/Unix_Tools/blob/master/images/touch.png?raw=1">

We can delete files by using the command,

- **<code>rm</code>** 

Let's delete <code>file3.txt</code> in the jupyter notebook,


```python
ls
```

    file1      file2      file3.txt



```python
rm file3.txt
```


```python
ls
```

    file1  file2


You can delete multiple files by just having a space between the files after <code>rm</code>. We can view the contents a file using the command,

- **<code>less</code>**

Let's take a look at the contents of <code>file1</code>:


```python
less file1
```

The results of typing this into the terminal are seen below:

<img src="https://github.com/mdh266/Unix_Tools/blob/master/images/less.png?raw=1">

We can scroll through the lines of the file by pressing the <code>enter</code> button.  We can exit the file by typing <code>q</code>.

Finally, let's show how to delete all a non-empty file.


```python
cd ..
```

    /Users/Mike/Documents/DS_Projects/Unix_Tools


If we just try to delete <code>Direc1/</code>, we'll get a warning that it is not empty and it won't delete the directory,


```python
rmdir Direc1/
```

    rmdir: Direc1/: Directory not empty



```python
ls
```

    [34mDirec1[m[m/           Unix_Tools.ipynb  [34mimages[m[m/
    README.md         file1


To delete this directory and everything inside we can use the command,

- **<code>rm -rf</code>**

The <code>-r</code>after the <code>rm</code> stands for recursive. We could also use <code>cp -r direc1 direc2</code> to copy all the contents of <code>direct1</code> to <code>direct2</code>.  The <code>-f</code> forces command to take place.  Let's try it:



```python
rm -rf Direc1/
```


```python
ls
```

    README.md         Unix_Tools.ipynb  file1             [34mimages[m[m/


It worked!  One last tool that for working with files in Unix that is extremely helpful is,


<a href="https://www.gnu.org/software/grep/manual/grep.html">grep</a>

grep is a utility that searches any given input files or directory, selecting lines that match one or more patterns using <a href="https://en.wikipedia.org/wiki/Regular_expression">regular expression</a>.

An example migh tbe to *search and list all the files in a directory that end with the word `.json`.* You can see an example where I do this on my Destkop below:


<img src="https://github.com/mdh266/Unix_Tools/blob/master/images/grep.png?raw=1">


There are lot more useful things that grep can do and for more examples see this <a href="http://www.thegeekstuff.com/2009/03/15-practical-unix-grep-command-examples">article</a>.

Now that we've got that under our belt lets see how we can edit files.  

## Vim
Vim (or <a href="https://en.wikipedia.org/wiki/Vi">Vi</a> as it is sometimes known as) is a file editor that is on every Unix/Linux and Mac computer by default. There is a historical debate between developers who perfer Vi vs. Emacs vs. some newer editor.  I think it's is important to know the basics of Vim, solely because on you may have to make a few changes on a machine where there are no other editors to use except vi.

If you type,

- **<code>vi</code>**


in to the terminal you can see that we have new window with <code>no name</code> displayed at the bottom of the screen; this means that we are working with an unnammed file.  To quit vi without saving the file (that we're editing) we first press the <code>esp</code> key followed by a colon (:) and then use,

- **<code>q!</code>**

We can create a file with a specific name and open it using vi using the following one line command,

    vi hello.py
    
This creates a file called <code>hello.py</code> as seen below:

<img src="https://github.com/mdh266/Unix_Tools/blob/master/images/hello.py1.png?raw=1">


We can edit the file by typing,

- **<code> i </code>**

for insert. We do this and write the following,


<img src="https://github.com/mdh266/Unix_Tools/blob/master/images/vi_saving.png?raw=1">


We can save the edits to the file using the command,

- **<code> w </code>**

which stands for *write*. We save our edits by pressing the <code>esp</code> key, then <code>:w</code>and hit enter.  You can scroll up and down lines in the file using the up and down arrows and scroll left or right across the screen using the left and right arrows. We can then exit editing the fiel in vi by using,

- **<code> q </code>**

by again first using the <code>esp</code> then <code>:q</code> (one could also save and quit by using <code>esp</code> then <code>:wq</code>).

We then can excute the contents of this file using the command from the terminal,

    python hello.py
    
We see the results below:


<img src="https://github.com/mdh266/Unix_Tools/blob/master/images/results.png?raw=1">


Now, let's go back and edit our <code>hello.py</code> file by adding the following on top:

<img src="https://github.com/mdh266/Unix_Tools/blob/master/images/hello.py2.png?raw=1">

And then exit and type <code>ls -al </code> into our terminal we will see the following for <code>hello.py</code>:


<img src="https://github.com/mdh266/Unix_Tools/blob/master/images/permissions1.png?raw=1">


You can see the <code>-rw-r--r--</code>.  The <code>r</code> stands that fact we have for *read* permissions to the file, the <code>w</code> stands for the fact we have *write* permissions to the file, and if we had a <code>x</code> that would mean we could *excute* the file. We can give ourselves the ability to execute the file using the command

- **<code>chmod</code>**


We'll simply give anyone access to read, write and execute this file by typing,

    chmod 777 hello.py
    
into the terminmal.  Now if you type <code>ls -al </code> you can see we have the ability to execute the file as well:

![](images/permissions2.png)

I've skipped a bunch of detiails, so for more information on chmod, see <a hre="https://en.wikipedia.org/wiki/Chmod">here</a>. We can npw execute the file by typing,

    ./hello.py

The results are shown below:

![](images/results2.png)

Note that if we did not add the line

    #!/usr/bin/env python
    
on to the top of the <code>hello.py</code> file it wouldn't be executable, to understand why this is so check out this <a href="https://stackoverflow.com/questions/2429511/why-do-people-write-usr-bin-env-python-on-the-first-line-of-a-python-script">post</a>.

There are tons more commands you can use in Vi (for more see <a href="https://docs.oracle.com/cd/E19683-01/806-7612/6jgfmsvqn/index.html">here</a>), but if you just looking to quickly open and edit a file this is pretty much all you need to know.  One reason why you would use Vi over another editor is that you may be working over a network and not have the ability to use any advance editors like <a href="https://www.sublimetext.com/3">sublime</a> which require a graphical interface.  This point segways directly into the next topic which is working over a network.

## Working Over A Network <a class="anchor" id="Network"></a>
-------------------

Working over a network to access a computer or cluster is something you will often have to do as a data scientist.  The basic way to "log on" to another unix computer is with using

-**<code>ssh</code>**

<a href="https://en.wikipedia.org/wiki/Secure_Shell">ssh</a> stands for secure shell and is a way to operate network services/computers securely over a non-secure network. The way you would access a computer with an address or ip address <code>computer_address</code> using your login name <code>login_name</code> is by the command,

    ssh login_name@computer_address
    
you'll then be prompted for your password and after typing it in you can hit <code>enter</code>.  Once on the computer/cluster you can nativate through directories and edit files as we learned above.  


Now, how do you get files back to your own computer from the cluster or put files from your computer on the cluster? There are many ways to do this, the easiest involve three things:

1. Creating a compressed archive
2. Transferring the archive
3. File extraction from the archive

To create an archive, or a one file collection of directory/folder, as well as compress and extract (uncompress) the archive we will use the,

- **<a href="http://linuxcommand.org/man_pages/tar1.html">tar</a>** 

command. We can create compressed version of the directory with the command,

    tar -cvzf compressed_file_name.tar.gz directory_name
    
To uncompress or extract the "tar file" we use the command,

    tar -xvzf compressed_file_name.tar.gz
   
We compress and uncompress the file/directories before and after transfering to reduce the amount of data that must be sent as well as the time it takes to transfer it. In order to get files to and from a cluster or machine we use,


- **<a href="https://en.wikipedia.org/wiki/SSH_File_Transfer_Protocol">sftp</a>**

which stands for **secure file transfer protocol.**  You can use <code>sftp</code> just like <code>ssh</code>,

    sftp login_name@computer_address
  
we can then nativate the cluster or computer using the commands we disucussed above.  Once we get to the directory with the file we can say,

    get file_name

If we want to get an entire directory we use the command, 

    get -r directory_name
   
The file(s) will be transfered to the our local machine in the directory we called <code>sftp</code> from. In order to transfer a file from your local macine to a remote machine we first <code>cd</code> to the directory that contains the file we wish to transfer and then call <code>sftp</code> from there.  We then write,

    put file_name
    
If we want to push an entire directory we use the command, 

    put -r directory_name

We can quit of <code>sftp</code> by typing <code>exit</code> from the command line. 

One last tool that is useful while working over a netwtork is,

- **<a href="https://www.gnu.org/software/screen/manual/screen.html">screen</a>**

Screen can be used to run programs (on a remote computer using ssh) even after you disconnected from the computer and closed the ssh session. You can also reattach to that sesssion at a later time and continue working from where you left off!  <a href="https://en.wikipedia.org/wiki/Nohup">Nohup</a> and <a href="https://en.wikipedia.org/wiki/Tmux">tmux</a> are also solutions for persisting programs if you are intested in other options. Screen can do much more than just persist a running program, but I won't cover that here.  Instead let's quickly go over how to create a session and then reattach that session. 

First we create a screen session by typing,

    screen
    
and we then get a terminal pretty much like the one we had before.  We can then run our program of choice and detach the screen session by holding the `control button`, the `a` button and the `d` button.  After we doing this, we then  see in our terminal,


<img src="https://github.com/mdh266/Unix_Tools/blob/master/images/detached.png?raw=1">


which tells us that the session has been detached.  The program we ran will still be running on the background, even if we close our ssh connection.  We can reattach the screen session by typing,

    screen -list 
    
and get a list of all the active screen sessions and their id numbers.  We find the session we want and re-attach to it using the command,

    screen -r <session_id>
 
Finally, we can kill the screen session by simply typing,
    
    exit
    
into the terminal.  You can see the example of me detaching and re-attaching to a screen session below:


<img src="https://github.com/mdh266/Unix_Tools/blob/master/images/detached-attached.png?raw=1">

## A Word On Working With Git <a class="anchor" id="Git"></a>
-------------------

The last thing that you need to know is how to work with <a href="https://en.wikipedia.org/wiki/Git">git</a>.  You don't have to use git, but you should be working with some type of version/source control.  Version control is essential and has saved me countless times.  However, version control is useless unless you work with it properly. This means commiting often and having meaningful messages when you commit.  It also means when you move files around that you do it within the git framework, i.e.

    git mv file_name directory_name
    
or 

    git rm file_name
 
instead of

    mv file_name directory_name

and 

    rm file_name
    
In order to learn about git and <a href="https://github.com/">github</a> I used a lot of the videos on the deal.ii <a href="http://www.math.colostate.edu/~bangerth/videos.html">website</a>.

## Conclusion And More Resources <a class="anchor" id="More"></a>

In this blog post we went over some of the basics of using command line tools.  There are a lot other things out there to learn.  Some things that I didnt cover which are important are,

- **<a href="https://www.tutorialspoint.com/unix/unix-environment.htm">environment variables</a>**

- **<a href="https://en.wikibooks.org/wiki/Bash_Shell_Scripting"> Bash Scritpts</a>**

- **<a href="https://en.wikibooks.org/wiki/Bash_Shell_Scripting"> bashrc files</a>**

You should definitely invest in using <a href="https://en.wikipedia.org/wiki/Package_manager">package managers</a> as they make your life much easier by installing and updating libraries and figure out dependencies between libraries. The package mananager I use is <a href="https://brew.sh/">homebrew</a> and for Python I use <a href="https://packaging.python.org/tutorials/installing-packages/">pip</a> and <a href="https://www.continuum.io/">Anaconda</a>.  That's it for now.

Happy hacking!
