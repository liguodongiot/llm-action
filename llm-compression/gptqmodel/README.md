


git clone git@github.com:liguodongiot/GPTQModel.git


git remote add upstream git@github.com:ModelCloud/GPTQModel.git


# 拉取原始仓库数据
git fetch upstream --tags

# 如果你的主分支不是叫master，就把前面的master换成你的名字，比如main之类
git rebase upstream/main

# 推送
git push

# 推送tags
git push --tags






git checkout -b dev-code-v2.0.0 v2.0.0

# 将新分支推送到远程仓库
git push -u origin dev-code-v2.0.0