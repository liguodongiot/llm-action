
- 参考：https://blog.csdn.net/sdujava2011/article/details/138312278

########

git clone git@github.com:liguodongiot/sglang.git


git remote add upstream https://github.com/sgl-project/sglang.git


# 拉取原始仓库数据
git fetch upstream --tags

# 如果你的主分支不是叫master，就把前面的master换成你的名字，比如main之类
git rebase upstream/main

# 推送
git push

# 推送tags
git push --tags



########


# 查看所有tag
git tag

# 4. 检出指定的 tag 到新分支
# 替换 'tag_name' 为你想要的 tag 名称

git checkout -b dev-code-0.4.1 0.4.1

# 将新分支推送到远程仓库
git push -u origin dev-code-0.4.1


