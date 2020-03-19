#run with: bash check.sh

#!/bin/sh
git status

find ./* -size +50M | cat >> .check
find ./* -size +50M | cat >> .gitignore

#check the .check file and ensure that all files shown are .npy or .xlsx
#these files are too large (>50M) for GitHub and need to be ignored
#if there are other files then they should probably be added to .gitignore
#https://github.com/sr320/course-fish546-2015/issues/43