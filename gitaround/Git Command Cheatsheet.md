### Git Command Cheatsheet



`git add`:add all current changes to the next commit.

`git status`:list which files are staged,unchanged,and un tracked.

`git reset`:reset staging area to match most recent commit,but leave the working directory unchanged.

`git reset --hard`:reset staging area and working directory to match most recent commit and overwrites all changes in working directory.

`git commit -m <message>`:commit the staged snapshot,but instead of launching a test editor,use <message>as the commit message.

`git branch`:list all of the branches in your repo.

`git branch -D <branch>`:

`git checkout <branch>` :checkout an existing branch.

`git checkout <commit>`:

`git checkout <branch or commit> -- <paths to file>`:

`git stash`:

`git stash pop`:

`git checkout -b <branch>`:create and check out a new baranch named <branch>.

`git init`:initialize the current directory as a git repo.

`git clone <URL>`:clone repo located at <URL> onto local machine.

`git remote -v`:list all currently configures remotes

`git remote add <name> <URL>`:add new remote repository,named <name>.

`git remote del <name>`:

`git push <host> <branch>`:

`git push -d <host> <branch>`:

`git pull <host> <branch>`:

`git fetch`:fetch all the remote fefs.

`git merge <branch>`:merge<branch>into the current branch.

`git cherry-pick <commit>`:

`git branch --set-upstream <host/branch>`：set tracking information for <host/branch>

