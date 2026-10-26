# 项目上传与日常修改指南

使用 VS Code、PowerShell 和已经安装的 Git。以用户亲手操作为主；先完成一个步骤、看懂输出，再进入下一步。本文命令是教程，不表示已经执行。

## 1. 先理解保存、提交和推送

- 保存（Ctrl+S）：将编辑器内容写入本机文件。
- Git：记录项目修改历史的工具。
- commit（提交）：把选择的文件修改记录为本地历史中的一个检查点。
- GitHub：保存远程 Git 仓库、便于备份和协作的网站。
- push（推送）：把本地提交上传到 GitHub。
- pull（拉取）：取回远程提交并整合到本地。

日常可以随时在 VS Code 改代码，不必先去 GitHub。保存不会自动上传；及时同步的方法是每完成一小块可理解的修改，就检查、提交、推送。

## 2. 首次初始化本地仓库

本项目已准备 `.gitignore`，排除虚拟环境、原始数据、生成图片、模型权重、令牌和本机配置。代码、教学规则、交接记忆和依赖清单将纳入版本管理。

```powershell
Set-Location -LiteralPath "E:\项目一" # 切换到项目根目录，确保 Git 操作针对这个项目。
git init -b main # 在当前文件夹创建本地 Git 仓库，并将初始分支命名为 main。
git status --short # 查看尚未提交的文件；此时 ?? 表示 Git 尚未跟踪该文件。
```

预期出现源码、Markdown、依赖清单和 `.gitignore`；不应出现 `.venv`、`data` 或 `outputs`。若这些目录出现，先停下，检查 `.gitignore` 的名称、位置和内容。`.gitignore` 排除文件参与版本管理，不会删除本地文件。

## 3. 设置提交身份

提交身份是历史记录中的作者信息，不等同于 GitHub 登录。以下设置只作用于当前项目。

先在 GitHub 的 [邮箱设置](https://github.com/settings/emails) 查看可用的 noreply 邮箱；也可以使用自己愿意出现在提交记录中的邮箱。不要照抄示例邮箱或虚构用户名。

```powershell
git config --local user.name (Read-Host "请输入提交记录中显示的名字") # 交互输入作者显示名，只写入当前仓库配置。
git config --local user.email (Read-Host "请输入 GitHub 设置中显示的 noreply 邮箱或你选择的邮箱") # 交互输入提交邮箱，不更改全局配置。
```

## 4. 创建第一次本地提交

```powershell
git add . # 将当前目录内未被忽略的新增和修改文件放入暂存区，准备形成一次提交。
git diff --cached --stat # 查看本次将要提交的文件和改动规模，确认包含代码、文档而没有数据或凭据。
git commit -m "Initial ACT learning checkpoint and shared project memory" # 将本次学习检查点保存到本地历史中。
```

暂存区就是“这次提交要包含哪些修改”的清单。第一次提交后，本地已有历史，但尚未上传。

## 5. 在 GitHub 创建一个空仓库

1. 登录 GitHub，打开 [创建仓库](https://github.com/new)。
2. 选择自己的账号作为 Owner，名称可用 `lerobot-smolvla-learning`，也可以自行命名。
3. 学习阶段建议先选 Private；如选择 Public，代码和个人学习记录会公开。
4. 本地已经有 README 和忽略规则，因此不要在网页端另外初始化 README、.gitignore 或许可证，以免初次同步时产生两条独立历史。
5. 点击 Create repository，复制仓库的 HTTPS 地址。实际仓库名称和账号以用户选择为准。

## 6. 连接远程仓库并首次推送

```powershell
$projectRepoUrl = Read-Host "请粘贴刚创建的 GitHub 仓库 HTTPS 地址" # 将实际地址保存在临时变量中，不需要把账号或令牌写入脚本。
git remote add origin $projectRepoUrl # 为本地仓库添加名为 origin 的远程仓库地址。
git remote -v # 核对远程地址，确认是自己的目标仓库。
git push -u origin main # 上传 main 分支，并设置后续 push 和 pull 使用的默认远程分支。
```

若 Git 打开登录窗口，在浏览器中完成 GitHub 授权。不要把令牌或密码贴进聊天；遇到权限或网络错误，贴回不含凭据的报错。没有收到推送成功结果前，不能认为已上传。

如果仓库已经有 origin，不要重复运行 remote add，应先查看 remote -v。若远程不是空仓库或提示历史冲突，先查看具体差异，不使用 force push 或 reset --hard 来覆盖它。

## 7. 以后怎样及时修改和同步

开始工作前：如果远程可能有其他电脑或助手提交的修改，先确认本地干净，再拉取。

```powershell
git status --short # 检查当前是否有未提交改动；有改动时先处理，不盲目拉取。
git pull --ff-only # 在本地没有未处理改动且远程历史可直接前进时更新项目；出现分叉会停下，不自动制造合并提交。
```

完成一小块修改后，先保存（Ctrl+S），检查变化并运行对应脚本。如果只是改说明文档，检查文档即可，无需重新运行训练。

```powershell
git status --short # 列出本次改动的文件。
git diff # 查看尚未暂存的具体修改，确认改动符合目的。
git add -- 07_act_forward.py PROJECT_MEMORY.md # 示例：只暂存本轮实际修改的脚本和进度文件；换成你这次修改的文件名。
git diff --cached --stat # 再查看这次提交包含哪些文件。
git commit -m "Explain ACT forward loss and update learning progress" # 保存一个说明修改目的的本地检查点；提交说明按实际内容调整。
git push # 将这个检查点同步到已经配置的远程分支。
```

`git diff` 不显示尚未跟踪文件的内容，已暂存改动用 `git diff --cached` 查看。遇到冲突或不认识的 Git 提示时保留现场，把输出交给助手解释，不强制覆盖。

## 8. 在 VS Code 中操作同一流程

Git 初始化并关联远程后，左侧 Source Control（源代码管理）可查看文件变化。点击文件看差异，点加号暂存，输入说明后 Commit，再 Push。按钮名称随版本或语言可能不同。

初学时，先分清 Commit 和 Push。Sync Changes 往往包含拉取与推送；了解这两步后再使用一键同步。无论使用按钮还是终端，操作的是同一个 Git 仓库。

## 9. 切换到 Claude Code 继续

在 VS Code 打开本项目目录，并将 Claude Code 的会话目录设为项目根目录。若已安装终端命令，可以这样启动；本教程不代替安装和登录步骤。

```powershell
Set-Location -LiteralPath "E:\项目一" # 切换到共同项目根目录，让 Claude Code 能发现项目说明。
claude # 启动已安装并完成登录的 Claude Code；找不到命令时先确认安装方式或使用 VS Code 中已有的 Claude Code 入口。
```

本项目 `CLAUDE.md` 使用官方支持的 @ 导入，读取 `AGENTS.md` 和 `PROJECT_MEMORY.md`，避免维护重复进度。此方式不要求 Windows 符号链接。可在支持的 Claude Code 版本中用 `/context` 检查加载情况。

开始时可以发送：

> 请读取本项目 CLAUDE.md、AGENTS.md 和 PROJECT_MEMORY.md，先复述当前学习进度、尚未掌握的知识和下一步。继续用中文详细补课，每行代码配中文注释，每次只让我做一小步。当前先完成 GitHub 上传与日常同步教学；完成后再恢复 ACT 损失、潜变量和参数更新的学习。

不同电脑需要 clone 或 pull 取得最新代码和记忆，并重建自己的 `.venv`；不要复制或上传原电脑的虚拟环境。`CLAUDE.md` 传递的是文件中的上下文，不会自动搬运另一个产品的完整聊天历史、登录状态或账户记忆。

同一份文件不要让两个正在运行的助手同时修改。切换前保存文件、查看差异、更新共同进度；需要远程交接时再提交和推送。

## 官方参考

- [GitHub：上传本地代码](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)
- [GitHub：忽略文件](https://docs.github.com/en/get-started/git-basics/ignoring-files)
- [VS Code：源代码管理入门](https://code.visualstudio.com/docs/sourcecontrol/quickstart)
- [Claude Code：项目记忆和文件导入](https://code.claude.com/docs/en/memory)
