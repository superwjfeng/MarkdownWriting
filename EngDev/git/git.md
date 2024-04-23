# Git使用

## *版本控制*

### 版本控制术语和概念

1. Codeline/Branch： Codeline是指软件开发过程中的一个代码分支或线路，它代表了软件的一个特定版本或变体。通常，在软件开发过程中，为了满足不同的需求或开发不同的功能，开发团队可能会创建多个并行的代码线。每个Codeline可能包含了特定的功能、修复或其他改动。例如，一个Codeline可以代表一个主要版本（如v1.0）、一个开发版本（dev）、一个修复分支（hotfix）等。
2. Baseline： Baseline是指在软件开发过程中的一个重要时间点或里程碑，用于标记一个Codeline在特定时间的状态。当软件开发到达一个稳定的阶段，或者需要进行发布或测试时，开发团队通常会将当前的Codeline标记为一个Baseline。Baseline可以看作是一个特定时刻的快照，它具有确定性，即在同一个Baseline下的代码和配置是一致的，能够被准确地复现和部署。
3. Mainline： Mainline是指软件开发中的主要代码线路或主要分支，它通常是整个开发过程中的核心分支。Mainline包含了主要的功能和改动，并且经常用于集成不同开发者的代码。Mainline通常是一个稳定的Codeline，它作为整个开发团队的主要代码库，用于构建软件的最新版本。

### 版本控制工具分类

* 集中式版本管理系统：版本库是几种放在中央服务器的，而工作的时候要先从中央服务器拉取最新版本，然后工作完成后再push到中央服务器。集中式版本控制系统必须要联网才能工作，对网络带宽要求较高
  * [Subversion](www.subversion.apache.org): open source, still widely used
  * [Perforce](https://www.perforce.com): proprietary, mostly enterprise use
  * [Concurrent Versions Systems (CVS)](https://www.nongnu.org/cvs/) – open source, no longer recommended for new projects

* 分布式版本控制系统：没有中央服务器，每个人本地就有一个完整的版本库
  * [Git](https://git-scm.com): open source, one of the most popular DVCS
  * [Mercurial](www.mercurial-scm.org): open source
  * [Darcs](http://darcs.net): darcs.net – open source
  * [BitKeeper](http://www.bitkeeper.org): started proprietary, now open source, influenced creation of Git

## *git分区*

### 工作区、暂存区和版本库

Git追踪管理的是修改，而不是文件

<img src="git三个区域.png" width="50%">

* 工作区 Working directory：放需要管理的代码和文件的目录
* 暂存区 Stage area/index：一般放在 `.git` 目录下的index文件中
* 版本库 Repository (locally)：`.git` 这个隐藏目录被称为Git的版本库

## *git 配置*

### git的设置

`git config`: Git 是一个 [高度可定制的](https://git-scm.com/docs/git-config) 工具。可以通过 `git config -l` 来查看设置列表

可以通过 `git config --unset` 来重制当前仓库的设置

`git config --global` 来设置当前host的全部仓库

## *基础指令*

Git 处理snapshot场景的方法是使用一种叫做 staging area 暂存区的机制，它允许用户指定下次快照中要包括那些改动

* `git help <command>`: 获取 git 命令的帮助信息
* `git init`: 创建一个新的 git 仓库，其数据会存放在一个名为 `.git` 的目录下
* `git status`: 显示当前的仓库状态
* `git add <filename>`: 添加文件到暂存区
* `git commit -m "YOUR COMMIT"`：创建一个新的commit，放到本地的repository中
* `git log`: 显示历史日志
* `git log --all --graph --decorate`: 可视化历史记录（有向无环图）
* `git diff <filename>`: 显示与暂存区文件的差异
* `git diff <revision> <filename>`: 显示某个文件两个版本之间的差异
* `git checkout <revision>`: 更新 HEAD 和目前的分支

### 撤销与回滚

* `git commit --amend`: 编辑提交的内容或信息
* `git reset [--soft | --mixed | --hard] <file>`: 用于回退 rollback。**本质是回退版本库中的内容**
  * `--mixed` 为**默认选项**，使用时可以不用带该参数。该参数将暂存区的内容退回为指定提交版本内容，工作区文件保持不变
  * `--soft` 参数对于工作区和暂存区的内容都不变，只是将版本库回退到某个指定版本
  * `--hard` 参数将暂存区与工作区都退回到指定版本。**切记工作区有未提交的代码时不要用这个命令**，因为工作区会回滚，没有commit的代码就再也找不回了，所以使用该参数前一定要慎重
  * file的说明
    * 可以直接使用commit id，表示指定退回的版本
    * 一般会使用HEAD来替代：HEAD 表示当前版本，`HEAD^` 上一个版本，`HEAD^^` 上上一个版本，以此类推。也可以使用 `~数字` 来替代，比如 `HEAD~0` 表示当前版本，`HEAD~1` 为上一个版本，依次类推
  * `git reset` 是可以来回rollback的，可以使用reflog来查看commit ID
* `git checkout -- <file>`：丢弃修改，让工作区中的文件回到最近一次add或commit时的状态
* `git restore`: git2.32版本后取代git reset 进行许多撤销操作

应用场景：撤销回滚的目的是为了防止自己写的代码影响到远程库这个公共空间中的代码

* 还没有add到暂存区
  * `git checkout -- <file>`，注意一定要带上 `--`，否则checkout就是用来对branch进行操作的
  * `git reset --hard file`
* 已经add到暂存区，但还没有commit到版本库中：`git reset [--mixed ｜ --hard] file`
* 已经commit到版本库中：前提是没有push到远程库 `git reset --hard file`

## *分支操作*

### 切换分支

移动HEAD指针来指向不同的分支指针，分支指针再指向不同的commit ID。分支指针都放在 `.git/refs` 下面，分成了本地的heads和远端的remotes

注意：在新建branch的时候是在当时的分支上进行了一次commit，即

```
--------+-----+-----+----------+-----+------+          Main Branch      
Last commit before new branch<-|     |->new commit due to new branch    
                                     |
                                     |------+          New Branch
```

* `git branch`：显示本地分支，`git branch -r` 查看远端分支。`git branch --set-upstream-to=origin/master`
  * `git branch <name>`: 创建分支
  * `git branch -d <BranchName>` 删除某条分支，注意删除某条分支的时候必须先切换到其他的分支上
  
* checkout切换分支

  * `git checkout <branch>`：切换到特定的分支

  * `git checkout -b <name>`：创建分支并切换到该分支。相当于 `git branch <name>; git checkout <name>`

  * checkout的使用前提：要保证当前分支上没有未提交的更改。如果有未commit的更改，需要先comimt它们，或者将它们暂存起来，即使用 `git stash` 将它们储藏起来，否则Git不允许切换分支以防止潜在的冲突。如果只是恢复文件，那么未提交的其他文件变动不会影响checkout操作

    否则就会报下面的错误

    ```
    error: Your local changes to the following files would be overwritten by checkout:
    	>>>>>>>>>>>> some changed files
    Please commit your changes or stash them before you switch branches.
    Aborting
    ```


* stash暂存未commit的内容：临时保存（或“暂存”）当前工作目录和索引（即暂存区）的修改状态，以便可以在一个干净的工作基础上切换分支或者做其他操作。这是一个非常有用的工具，尤其是当不想通过提交就保存当前进度的时候

  ```cmd
  git stash
  git stash save "Your stash message" # 等价于 等价于 git stash
  git stash list # 列出所有暂存的进度
  git stash apply # 应用最近的暂存进度
  git stash apply stash@{n} # 应用特定的暂存进度，其中 n 对应 git stash list 显示的暂存序号
  git stash drop # 删除最近的暂存进度
  git stash clear # 应用并删除最近的暂存进度
  git stash pop # 清空所有暂存进度
  git stash branch <branchname> stash@{n} # 创建一个新的分支并应用某个暂存进度，这条命令很好用，如果有冲突的话需要手动解决
  # 但是如果同名的远端branch已经存在的话建议不要用，因为会创建一个同名的本地分支，而stash的内容会覆盖branch上同名的文件，而远端的内容可能是不一样的
  ```

* 从Git 2.23版本开始，Git引入了新的`git switch`和`git restore`命令，以提供更符合直觉的方式来分别处理分支切换和文件恢复的行为

### 合并分支

* `git merge <revision>`：合并到当前分支
* `git mergetool`: 使用工具来处理合并冲突
* `git rebase <basename> <topic_name>`: 将一系列补丁 topic_name 变基 rebase 到新的基线 basename

### 关于合并冲突的问题

合并冲突模式

* Fast-forward：看不出来是否有创建分支并merge，看起来就像是直接在当前分支上修改的，可以通过 `git merge [--no-ff -m "提提交信息"] <branch>` 来不使用fast-forward模式来merge，注意 -m 是一定要写的
* No-ff 非fast-forward：可以看出来有没有经过merge

```
<<<<<<< HEAD
当前分支的内容
=======
其他分支上发生冲突的内容
>>>>>>>
```

解决冲突的方式是把不需要的代码全部删除，包括尖括号提示符

merge冲突需要手动解决，并且**merge后一定要再进行一次commit**。HEAD会指向merge的新提交，但被merge的分支仍然会指向自己原来的commit

```
*   commit 1f7605d4cf4e180c21b693f2aed0f945611fa33b (HEAD -> main)
|\  Merge: 27b9c7b 4dcd274
| | Author: wjfeng <wj.feng@tum.de>
| | Date:   Sat Jun 17 16:36:45 2023 +0200
| |
| |     main branch conflict fixed
| |
| * commit 4dcd27411054788c7030e7936d62ebb1ca2a3247
| | Author: wjfeng <wj.feng@tum.de>
| | Date:   Sat Jun 17 16:33:19 2023 +0200
| |
| |     test branch on dev
| |
* | commit 27b9c7bbc515a114af67ce45c5fa86d0e5591765
|/  Author: wjfeng <wj.feng@tum.de>
|   Date:   Sat Jun 17 16:34:25 2023 +0200
|
```

从main上创建了一条新的branch后，若main上没有新的commit或者没有冲突就可以直接fast-forward merge

### 分支管理的原则

* master/main：稳定分支，用来发布新版本。不允许直接修改这个分支
* dev：不稳定的开发分支，用来开发新功能。等测试完毕后再merge到master分支
* 用 `git stash` 命令将当前工作区已经add（被git追踪了）但是还没有commit的内容存起来，会放在 `.git//refs/stash` 临时存储区中，将来可以被恢复。不能把没有add的文件stash
* 可以通过 `git stash list` 来查看临时存储区的内容

将代码merge到master中的好习惯是：**在master上merge完修复好的bug后，先切换到dev上merge master，再切换到master上merge dev**。而不是在master上merge dev。在merge的过程中也有可能因为一些误操作（少merge多merge了一行）等原因而出错，因此若直接在master上merge dev出bug了，master分支会受到直接的影响，而在dev上merge master，就算出错影响的也只是dev

## *rebase vs. merge*

[https://vue3js.cn/interview/git/git%20rebase_%20git%20merge.html#二、分析](https://vue3js.cn/interview/git/git%20rebase_%20git%20merge.html#二、分析)

### merge

<img src="merge.drawio.png" width="40%">

merge 过程还是很直观的，且 merge 是一种非破坏性的操作，对现有分支不会以任何方式被更改，保留了完整不过可能会比较复杂的历史记录

### rebase

<img src="rebase.drawio.png" width="40%">

1. rebase 会找到不同的分支的最近共同祖先，比如说上面的 B
2. 然后对比当前分支相对于该祖先的历次提交，提取相应的修改并存为临时文件。注意⚠️：老的提交 X 和 Y 也没有被销毁，只是单纯地不能再被访问或者使用了
3. 然后临时文件向前移动的时候如果发生冲突就需要解决冲突
4. rebase 之后，master 的 HEAD 位置不变。因此要合并 master 分支和 bugfix 分支

rebase有以下优势

1. **线性提交历史**：rebase操作会将当前分支的提交移到目标分支的最后面，从而使提交历史更加线性清晰
2. **减少合并提交**：rebase会避免创建额外的合并提交，使提交历史更加整洁

## *远程库*

### 远端操作

远端merge有两种方法，一种是在本地merge后push到远端，另一种是使用Pull request，Pull requset是给仓库管理员看的合并请求，实际开发中推荐使用PR

当用户在自己的分支上进行代码修改后，用户可以向主项目（或主分支）提交一个pull request。这相当于提议将用户的代码更改合并到主项目中。Pull request允许其他开发者对用户的代码进行审查、提出修改建议或进行讨论。通过pull request，团队成员可以共同讨论代码变更、解决问题并确保代码质量

**从远程仓库克隆后，Git会自动把本地的master分支和远程的master分支连接起来，并且远程仓库的默认名称是origin**

创建远端分支有两种方法，要么直接在远程库手动创建，要么在本地push一个新的分支。**绝对禁止直接在远程库中修改代码，必须是在本地修改后将改变push到远程库**

* `git clone`：从远端下载仓库

* `git remote -v`：列出远端

* `git remote add <name> <url>`：**添加一个远端**，其中 `<name>` 一般就是origin

* `git push <remote repo> <local branch>:<remote branch>`：将本地的某一个分支传送至远端的某一个分支并更新远端引用

  若本地分支和远程分支同名，可以将 `git push origin master:master` 省略冒号成 `git push origin master`

* `git fetch`：从远端获取对象/索引，`git fetch` 是用来获取远端的更新，不会将远端代码pull到本地

* `git pull <remote repo> <remote branch>:<local branch>`：相当于 `git fetch + git merge`

  若是要让远程分支与当前分支合并，可以省略冒号后的内容

### 本地库与远端库的连接关系

可以通过 `git branch -vv` 来查看本地库与远程库的连接关系

常用情景：**拉取一个本地不存在的新的远程分支到本地成为一个新的分支** `git checkout -b <local-branch> origin/<remote-branch>`，通过这个命令可以自动建立本地与远程的分支连接，这样以后就可以直接用短命令 `git push` 了

但是若没有使用 `git checkout -b <local-branch> origin/<remote-branch>`，或者说忘记打后半部分了导致没有建立连接关系也不用紧，还可以使用`git branch --set-upstream-to=<remote>/<remote branch>` 来创建本地和远端分支的关联关系

### Detached HEAD

当用户在 Git 中切换到一个特定的commit，而不是分支时， HEAD 引用会进入 detached HEAD 状态。这种状态下的提交可能会更加容易丢失，因此在进行任何修改之前，应谨慎考虑并理解当前所处的状态

在 detached HEAD 状态下，HEAD 直接指向一个具体的提交，而不是一个具名的分支。这意味着当前不再位于任何分支的最新提交上

当用户处于 detached HEAD 状态时，用户可以查看提交的内容，甚至进行修改和提交新的更改，但这些更改将不会与任何分支相关联。若在 detached HEAD 状态下进行提交，这些提交可能会很难恢复，因为没有分支引用指向它们

Detached HEAD 状态通常发生在以下几种情况下

1. 使用 `git checkout` 命令切换到特定的提交，而不是分支名称
2. 使用 `git checkout` 命令切换到一个 tag
3. 使用 `git checkout` 命令切换到一个远程分支的具体提交

要解决detached HEAD状态，可以执行以下操作之一

1. 若是意外进入了detached HEAD状态，但没有进行任何修改，可以直接通过运行 `git switch -` 或 `git checkout -` 返回到之前所在的分支
2. 若是在detached HEAD状态下进行了修改，并且希望将这些更改与一个新的分支关联起来，可以使用 `git branch <new-branch-name>` 来创建一个新的分支，然后使用 `git switch <new-branch-name>` 切换到新的分支，并commit更改

## *子模块*

Git Submodule 是一个将其他 Git 仓库作为子目录嵌入到当前 Git 仓库中的功能。如果项目依赖于第三方代码或者其他组件，可以使用 submodule 来管理这些外部资源

### 添加 Submodule

1. **添加新的submodule**：要在你的项目中添加新的 submodule，你需要使用 `git submodule add` 命令，后面跟着仓库地址和希望将该仓库添加到的路径。

   ```cmd
   git submodule add <repository> <path>
   ```

   例如：

   ```cmd
   git submodule add https://github.com/user/repo.git external/repo
   ```

2. **初始化 submodule**：如果你克隆了一个包含 submodules 的项目，则需要初始化 submodule。

   ```cmd
   git submodule init
   ```

3. **更新 submodule**：初始化之后，使用以下命令更新 submodule，以获取其内容。

   ```cmd
   git submodule update
   ```

### 克隆包含 Submodule 的仓库

**当要克隆一个包含 submodules 的仓库时，submodules 不会自动被克隆**。需要通过下面的命令进行初始化和更新：

```cmd
git clone --recurse-submodules <repository>
```

如果已经克隆了不带 submodules 的仓库，可以执行下面的命令来拉取 submodules：

```cmd
git submodule update --init --recursive
```

### 检出 Submodule 特定版本

如果想要检出 submodule 的特定提交或标签，你需要进入到 submodule 目录中，然后像在普通的 Git 仓库一样切换分支或检出提交：

```cmd
cd <submodule_path>
git checkout <tag_or_branch_or_commit>
```

完成以上操作后，应该在主仓库中做个提交来记录 submodule 的状态更改

### 对所有模块做相同的操作

`git submodule foreach` 能在每一个子模块中运行任意命令。比如说下面的命令 

```cmd
git submodule foreach --recursive git reset --hard
```

### 拉取所有 Submodule 的最新变更

如果想要更新项目中所有submodule到最新提交，可以使用下面的命令：

```cmd
git submodule update --remote
```

这个命令会将每个submodule都更新到它们所跟踪分支（默认是 master）的最新提交

### 提交和推送包含 Submodule 的变更

当在submodule中做了变更并且提交这些变更后，在主仓库也会看到submodule有变更。此时需要在主仓库做一个新的提交以跟踪submodule的新状态

然后，正常地推送主仓库和submodule仓库：

```cmd
git push --recurse-submodules=on-demand
```

这种方式确保同时推送主仓库和submodule的变更

### 删除 Submodule

删除submodule稍微复杂一些，需要几步操作：

1. 删除submodule相关的配置信息：

   ```cmd
   git submodule deinit -f <path_to_submodule>
   git rm -f <path_to_submodule>
   ```

2. 从 `.gitmodules` 文件和 `.git/config` 移除submodule配置

3. 如果需要的话，手动从工作树中删除 submodule 目录：

   ```cmd
   rm -rf .git/modules/<path_to_submodule>
   ```

4. 提交这些变更到主仓库

以上就是Git Submodule的一些基础用法。Submodule能够很好地用于管理和维护项目依赖，但它增加了项目管理的复杂性，尤其是对于新手来说。因此，在开始使用前要仔细考虑是否真的需要它

# Git原理

```cmd
# 检出 Git 源码
$ git clone https://github.com/git/git.git
$ cd git
# 切换到第一次提交
$ git checkout e83c5163316f89bfbde7d9ab23ca2e25604af290
```

## *Git文件系统*

## *Git的对象模型*

```cmd
$ mkdir GitTest
$ cd GitTest
$ git init
```

<img src="git版本库内容.png" width="35%">

Object是git管理的最基本的单元，object代表了一个普通的文件。每一个被git管理的文件都会计算出一个hash值然后压缩放置于 `.git/objects` 目录中。修改的工作区内容的索引会写入对象库的一个新的git对象 object 中

一共有四个Object类型

```
CommitObject  = 1 // 提交
TreeObject    = 2 // 目录
BlobObject    = 3 // 文件
TagObject     = 4 // 标签
```

```pseudocode
type Object interface {
	ID() plumbing.Hash // 哈希值
	Type() plumbing.ObjectType // 类型
	Decode(plumbing.EncodedObject) error // 写入
	Encode(plumbing.EncodedObject) error // 读取
}
```







Git 将顶级目录中的文件和文件夹作为集合，并通过一系列快照 snapshot 来管理其历史记录。每一个文件被称为Blob对象，相当于是字节Array数据对象，目录被称为Tree，它将名字String于Blob对象或另外的树映射 `map<string, object>`

```
<root> (tree)
|
+- foo (tree)
|  |
|  + bar.txt (blob, contents = "hello world")
|
+- baz.txt (blob, contents = "git is wonderful")
```

Git使用由snapshot组成的有向无环图 directed acyclic graph 来建模历史。有向无环图的意思是每一个snapshot都有一系列的parent

每一个snapshot称为一个commit，每一个snapshot都会会指向它之前的snapshot。用伪代码可以表示成

```
// 文件就是一组数据
type blob = array<byte>

// 一个包含文件和目录的目录
type tree = map<string, tree | blob>

// 每个提交都包含一个父辈，元数据和顶层树
type commit = struct {
    parent: array<commit>
    author: string
    message: string
    snapshot: tree
}
```

### 对象和内存寻址

Git 中对象根据内容地址寻址，在储存数据时，所有的对象都会基于它们的SHA-1 哈希值进行寻址

```
objects = map<string, object>

def store(object):
    id = sha1(object)
    objects[id] = object

def load(id):
    return objects[id]
```

Blobs、树和提交都一样，它们都是对象。当它们引用其他对象时，它们并没有真正的在硬盘上保存这些对象，而是仅仅保存了它们的哈希值作为引用

但是哈希值是记不住了，所以要给他们起别名，也就是建立一个用string表示的reference 引用。和C++不同，这里reference应该被理解为指针，它可以不断变动指向不同的哈希值（或者说commit）。这样，Git 就可以使用诸如 “master” 这样人类可读的名称来表示历史记录中某个特定的提交，而不需要在使用一长串十六进制字符了

```python
references = map<string, string>

def update_reference(name, id):
    references[name] = id

def read_reference(name):
    return references[name]

def load_reference(name_or_id):
    if name_or_id in references:
        return load(references[name_or_id])
    else:
        return load(name_or_id)
```

git的分支实质上仅是包含所指对象的SHA-1校验和文件，所以它的创建和销毁都很快。创建一个新分支就相当于网一个文件中写了41个字节



## *Reference*

Git 支持三种引用类型，不同的引用类型对应的引用文件各自存储在 `.git/refs/` 下的不同子目录中。

* **HEAD 引用**
* **标签引用**
* **远程引用**

### 一些特殊的reference

* 当前的位置特殊的索引称为 HEAD
* origin一般用作本地对remote repository的名称，它是 `git clone` 时的默认remote库名称，可以 `git clone [-o RemoteName] ` 换一个名字
* 本地 `git init` 时的默认branch名称是master。因此对远程库的本地branch名称是，`<remote>/<branch>`，即origin/master









分析一下这些步骤

Cloning into 'tool/build_management/cache/devcar/mf_system'...
Username for 'https://devops.momenta.works': weijian.feng@momenta.ai
Password for 'https://weijian.feng@momenta.ai@devops.momenta.works':
remote: Azure Repos
remote: Found 896710 objects to send. (14539 ms)
Receiving objects: 100% (896710/896710), 488.88 MiB | 22.78 MiB/s, done.
Resolving deltas: 100% (384322/384322), done.











