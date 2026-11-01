# Git 常用命令：结合本项目学习

这是一份参考表，不是要求本轮全部执行的命令清单。所有命令都在正确的仓库目录下运行；本项目为 `E:\项目一`。实际上传状态见 `PROJECT_MEMORY.md`。

## 1. 先理解四个位置

工作目录 → 用 git add 选择修改 → 暂存区 → 用 git commit 保存版本 → 本地仓库 → 用 git push 上传提交 → GitHub 远程仓库。

- 工作目录：VS Code 中正在编辑的实际文件。Ctrl+S 保存到这里。
- 暂存区：这次提交准备包含的内容。它可以与工作目录不一样。
- 本地仓库：电脑上的版本历史，保存在 `.git` 中；提交后会得到类似 `436f4ff` 的编号。
- 远程仓库：GitHub 上保存的提交和分支。它不会自动收到你刚保存的文件。

例子：给 `07_act_forward.py` 加注释并保存，只改变工作目录；暂存后才进入这次提交清单；提交后成为本地历史；推送成功后才出现在 GitHub。

## 2. 先熟悉这些查看命令

以下命令只用于查看，不会修改脚本内容。

```powershell
git status # 查看当前分支，以及哪些文件未跟踪、已修改、已暂存。
git status -sb # 用短格式显示文件状态，同时显示分支信息。
git diff # 比较工作目录与暂存区，查看已跟踪文件尚未暂存的改动。
git diff --cached # 比较暂存区与最近一次提交，查看这次准备提交的具体改动。
git diff --cached --stat # 只显示准备提交的文件和改动规模，适合先看总体范围。
git log --oneline -5 # 查看最近五条本地提交的简短编号和说明。
git show --stat HEAD # 查看当前分支最新提交包含哪些文件改动。
git remote -v # 查看远程仓库的简称及其地址。
git branch -vv # 查看本地分支、所在提交及已设置的上游分支。
```

`HEAD` 通常指向当前所在分支的最新提交。`git diff` 不显示未跟踪文件的内容；使用 status 可以发现它们。远程跟踪信息反映上次联系远程时的记录，不是持续联网的实时状态。

## 3. 日常改代码最常用的流程

先在 VS Code 修改、Ctrl+S 保存，运行与修改有关的检查，再选择文件。下面以实际改过脚本和项目记忆为例，提交说明也应根据真实修改调整。

```powershell
git status # 确认本轮究竟修改了哪些文件。
git diff # 阅读尚未暂存的修改，确认符合本轮实验目的。
git add -- 07_act_forward.py PROJECT_MEMORY.md # 只将指定脚本和记忆文件的当前修改放入暂存区。
git diff --cached # 复查这次提交将包含的具体内容。
git commit -m "Explain ACT loss and update learning progress" # 创建本地提交，-m 后面的文字说明这次为什么修改。
git push # 在已经成功设置上游的情况下，将本地新提交上传到 GitHub。
```

`git add .` 会选择当前目录及子目录内未被忽略的新增、修改和删除；只提交部分文件时，按名字 add 更容易控制范围。文件暂存后若又继续编辑，新增的那部分修改需要再次 add 才会进入提交。

`git commit` 只保存已经暂存的内容；`git push` 上传提交，不会顺带上传尚未提交的修改。`nothing to commit` 表示没有符合本次提交条件的改动，需要结合 status 判断，而不是反复运行 commit。

## 4. 初次连接仓库与上传

`origin` 是远程仓库的常用简称；`main` 是本项目当前的分支名。它们不是邮箱或登录账号。本项目已经添加过 origin，后续不要重复运行 remote add。

```powershell
git push -u origin main # 首次将本地 main 推送到 origin，并把远程分支设为本地 main 的上游。
```

推送成功后，通常以后只需要 git push。初次成功输出可能包含 `[new branch] main -> main` 和上游跟踪设置；看到 commit 的输出本身不能证明 push 已成功。

登录授权、网络连接、权限或远程历史问题可能导致推送不能完成，应根据实际结果排查。LF/CRLF 是换行符转换提示；本项目此前的提示没有阻止本地提交。

## 5. 从 GitHub 取回修改

另一台电脑或另一个助手已经推送修改时，需要取回这些提交。先处理当前未提交的工作，不能把 pull 当作单纯下载覆盖文件。

```powershell
git fetch origin # 更新本地保存的远程分支记录，不自动合并到当前分支。
git pull --ff-only # 获取远程修改，并在历史可以直接前进时更新当前分支；历史分叉时停止。
```

当前尚未成功设置上游时，需要先完成首次连接和推送流程。若出现历史分叉或冲突，保留报错，先检查双方修改。

## 6. 撤销一次暂存，保留写好的代码

若某个文件不想放进本次提交，可以撤销它的暂存。本项目已经有初始提交，下面的写法适用。

```powershell
git restore --staged -- 07_act_forward.py # 将该文件的暂存内容恢复为 HEAD 版本，但保留工作目录中的代码修改。
```

执行后再看 status 和 diff，理解修改被放回了哪个位置。这里的 `--staged` 很关键，本条命令的目的是调整提交清单。

## 7. 后面再练习：分支与克隆

分支让我们给不同的实验记录不同的提交历史。它仍使用当前工作目录，不等于复制了一套文件夹；切换前先检查未提交修改。

```powershell
git switch -c act-loss-experiment # 从当前提交新建并切换到一个动作损失实验分支；名称需尚不存在。
git switch main # 切换回已经存在的 main 分支。
```

另一台电脑首次取得项目时，在准备存放项目的父目录执行 clone。当前已有项目的文件夹不需要重复克隆。

```powershell
git clone https://github.com/zbzb686/01.git # 下载仓库及其提交历史，默认创建名为 01 的文件夹。
```

克隆不会取得被忽略的 `.venv` 和原始数据；按 README 重建环境，并让脚本按固定版本读取数据。共同记忆、CLAUDE.md、教学规则会随已经提交和推送的文件一起传递。

## 8. 判断是否理解

1. 只保存脚本但未 add，它在哪个位置发生变化？
2. 已经 commit、尚未 push，GitHub 能看到这个版本吗？
3. 暂存后又修改脚本，直接 commit 是否包含最后那部分修改？

这些题可以逐个练习，不需要一次掌握所有命令。

## 官方参考

- [Git 入门教程](https://git-scm.com/docs/gittutorial)
- [status 与短格式状态](https://git-scm.com/docs/git-status)
- [push 与上游分支](https://git-scm.com/docs/git-push)
