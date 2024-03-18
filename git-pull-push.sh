git pull origin main
git add .

#time=`date -Iminutes`
time=`date +"%Y-%m-%d_%H:%M:%S"`

echo $time

commit_info="update-""$time"

git commit -m $commit_info

git push origin main



