## *zsh*

### iTerm2

iTerm2 是 Terminal 的替代品，也是 iTerm 的继任者。它适用于装有 macOS 10.14 或更高版本的 Mac

## *Windows*

### Windows系统上的终端 & shell工具

* Command Prompt（命令提示符）：是Windows内置的原始命令行工具，提供基本的文件和系统管理命令
  
  可通过在开始菜单中搜索“cmd”或“命令提示符”来打开。支持一系列命令，如`cd`（改变目录）、`dir`（列出目录内容）等
  
* PowerShell：是一个更强大和灵活的命令行工具，支持脚本编写和更高级的系统管理任务。使用类似于Linux的命令风格，支持对象管道、脚本编写等功能
  
* Windows Terminal：一个现代化的终端应用程序，支持同时打开多个标签页，可自定义外观和行为。支持不同的终端引擎，如Command Prompt、PowerShell、Linux子系统等

### Powershell 命令

* Get-Help：用于获取有关PowerShell命令的帮助信息。例如，`Get-Help Get-Process` 将提供有关获取进程信息的帮助
* Get-Command：用于获取系统中可用的命令列表。例如，`Get-Command` 将列出所有可用的PowerShell命令
* Get-Process：获取正在运行的进程的信息。例如，`Get-Process` 将列出所有正在运行的进程
* Get-Service：获取系统中的服务信息。例如，`Get-Service` 将列出所有安装的服务
* Get-EventLog：用于获取事件日志的信息。例如，`Get-EventLog -LogName System` 将显示系统事件日志的内容
* Get-Item：获取文件或文件夹的属性。例如，`Get-Item C:\Example\File.txt` 将显示文件的属性
* Set-ExecutionPolicy：用于设置脚本的执行策略。例如，`Set-ExecutionPolicy RemoteSigned` 将允许本地脚本的执行，但要求远程脚本有数字签名
* Start-Process：启动一个新的进程。例如，`Start-Process "notepad.exe"` 将启动记事本应用程序
* 文件、文件夹相关命令
  * Invoke-Item：打开某个路径的文件夹
  * New-Item：创建新的文件、文件夹或注册表项。例如，`New-Item -ItemType Directory -Path C:\Example` 将创建一个名为Example的新文件夹
  * Remove-Item：删除文件或文件夹。例如，`Remove-Item C:\Example\File.txt` 将删除指定的文件
  * Copy-Item：复制文件或文件夹。例如，`Copy-Item C:\Source\File.txt C:\Destination` 将复制文件到目标位置
  * Get-Content：读取文件的内容。例如，`Get-Content C:\Example\File.txt` 将显示文件的文本内容
  * Set-ItemProperty：设置对象的属性。例如，`Set-ItemProperty -Path C:\Example\File.txt -Name "ReadOnly" -Value $true` 将设置文件的只读属性为真

### Powershell配置

Powershell的配置工具在 `C:\Users\Weijian Feng\Documents\PowerShell\Microsoft.PowerShell_profile.ps1`

可以用 `vim $PROFILE` 打开

```cmd
PowerShellGet\Install-Module posh-git -Scope CurrentUser -Force
Install-Module PSReadLine
```

```
# 上下方向键箭头，搜索历史中进行自动补全
Set-PSReadLineKeyHandler -Key Tab -Function MenuComplete # Tab键会出现自动补全菜单
Set-PSReadlineKeyHandler -Key UpArrow -Function HistorySearchBackward
Set-PSReadlineKeyHandler -Key DownArrow -Function HistorySearchForward

# git的自动补全
Import-Module posh-git

# 一些常用的别名
Set-Alias ll Get-ChildItem
Set-Alias vim nvim # 确保已经安装了nvim
```

### 权限问题

PowerShell 默认可能会禁止执行未签名的脚本，以提供额外的安全性。错误信息表明你遇到了执行策略问题，因为系统策略阻止了 `Microsoft.PowerShell_profile.ps1` 脚本的执行。

你可以通过更改 PowerShell 的执行策略来允许脚本执行。最常用的执行策略有：

- **Restricted**：不允许任何脚本运行。
- **AllSigned**：只运行由可信发行者签名的脚本。
- **RemoteSigned**：运行本地脚本和由可信发行者签名的远程脚本。
- **Unrestricted**：运行所有脚本（这是最不安全的选项）

要查看当前的执行策略，可以使用：

```
Get-ExecutionPolicy
```

若要将执行策略更改为 RemoteSigned（通常是推荐选项），允许本地脚本执行和已签名的远程脚本执行，可以使用以下命令：

```
Set-ExecutionPolicy RemoteSigned
```

执行上述命令时，你需要以管理员身份运行 PowerShell。如果你不是管理员或者希望只对当前用户更改执行策略，可以使用 `-Scope` 参数指定：

```
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### oh-my-posh

虽然叫做 oh-my-posh，但它不仅仅只能用于 Powershell，其他的shell也能用

[Introduction | Oh My Posh](https://ohmyposh.dev/docs/)

* 安装

  ```cmd
  winget install JanDeDobbeleer.OhMyPosh -s winget
  $env:Path += ";C:\Users\user\AppData\Local\Programs\oh-my-posh\bin" # 添加到环境变量
  ```

* 设置主题

  ```
  Get-PoshThemes # 查看系统上可用的主题
  ```

  `vim $PROFILE` 后再里面写上

  ```
  oh-my-posh init pwsh --config "$env:POSH_THEMES_PATH/kushal.omp.json" | Invoke-Expression
  ```

  ```
  . $PROFILE
  ```

### Nerd font

[Nerd Fonts - Iconic font aggregator, glyphs/icons collection, & fonts patcher](https://www.nerdfonts.com/#home)

Nerd Fonts 是一个开源项目，提供了一系列字体，专门设计用于在终端中显示图标和各种编程语言的特殊字符。这些字体包含了许多常见的开发者工具、图标和表情符号，使其在代码编辑器、终端和其他文本界面中更容易阅读和使用

带有 Windows Compatible 后缀的字体，专门为 Windows 系统优化过

带有 Mono 后缀的字体，所有字符大小相同，不带 Mono 后缀的字体，图标字符比常规字符大 1.5 倍

推荐 ComicShannsMono

win中添加字体的方式是将下载的字体解压缩后右键安装即可

## *tmux*