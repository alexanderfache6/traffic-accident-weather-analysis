#run with: bash check.sh

#!/bin/sh
git status

find ./* -size +50M | cat >> .check

#check the .check file and ensure that all files shown are .npy or .xlsx
#these files are too large (>50Mb) for GitHub and need to be ignored
#if there are other files then they should probably be added to .gitignore
