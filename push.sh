#run with: bash push.sh

#!/bin/sh

git add .

echo "Commit Message:"
read message
echo "Git Message: $message"

git commit -m "$message"

git push origin master