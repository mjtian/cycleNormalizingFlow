### Git Command Cheatsheet



`git add`:add all current changes to the next commit.

`git status`:list which files are staged,unchanged,and un tracked.

`git reset`:reset staging area to match most recent commit,but leave the working directory unchanged.

`git reset --hard`:reset staging area and working directory to match most recent commit and overwrites all changes in working directory.

`git commit -m <message>`:commit the staged snapshot,but instead of launching a test editor,use <message>as the commit message.

`git branch`:list all of the branches in your repo.

`git branch -D <branch>`:delete a local branch.

`git checkout <branch>` :checkout an existing branch.

`git checkout <commit>`:switch branches or restore working tree files.work on the top of <commit>

`git checkout <branch or commit> -- <paths to file>`:check out paths to the working tree,checkout to the <branch or commit>. 

`git stash`:stash the changes in a dirty working directory away.

`git stash pop`:remove a single stashed state from the stash list and apply it on top of the current working tree state.

`git checkout -b <branch>`:create and check out a new baranch named <branch>.

`git init`:initialize the current directory as a git repo.

`git clone <URL>`:clone repo located at <URL> onto local machine.

`git remote -v`:list all currently configures remotes.

`git remote add <name> <URL>`:add new remote repository,named <name>.

`git remote del <name>`: <name> is deleted. 

`git push <host> <branch>`:update remote refs along with associated objects.

`git push -d <host> <branch>`:all listed refs are deleted from the remote repository.

`git pull <host> <branch>`:fetch from and integrate with another repository or a local branch.

`git fetch`:fetch all the remote fefs.

`git merge <branch>`:merge<branch>into the current branch.

`git cherry-pick <commit>`:apply the changes introduced by some existing commits.<>is a more complete list of ways to spell commits.

`git branch --set-upstream <host/branch>`：set tracking information for <host/branch>

