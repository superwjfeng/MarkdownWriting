# Azure Pipelines

[Azure Pipelines 文档 - Azure DevOps | Microsoft Learn](https://learn.microsoft.com/zh-cn/azure/devops/pipelines/?view=azure-devops)

> Azure Pipelines 自动生成和测试代码项目。 它支持所有主要语言和项目类型，并结合了[持续集成](https://learn.microsoft.com/zh-cn/azure/devops/pipelines/get-started/what-is-azure-pipelines?view=azure-devops#continuous-integration)、[持续交付](https://learn.microsoft.com/zh-cn/azure/devops/pipelines/get-started/what-is-azure-pipelines?view=azure-devops#continuous-delivery)和[持续测试](https://learn.microsoft.com/zh-cn/azure/devops/pipelines/get-started/what-is-azure-pipelines?view=azure-devops#continuous-testing)，可以生成、测试代码并将其交付到任何目标

## *创建管道*

在repo的根目录中使用名为 `azure-pipelines.yml` 的 YAML 文件来定义管道

YAML各个字段的意义可以参考：[YAML schema reference | Microsoft Learn](https://learn.microsoft.com/en-us/azure/devops/pipelines/yaml-schema/?view=azure-pipelines)

- `trigger`：当 Git 仓库的`master`分支有新的提交时，该流水线即会被触发；

- `pool`：该流水线将会在一台 Linux 节点上运行，节点采用的镜像为`ubuntu-latest`；

- jobs：job的集合，里面还有很多job

- steps：pipeline执行的一步

  ```
  steps: [ task | script | powershell | pwsh | bash | checkout | download | downloadBuild | getPackage | publish | template | reviewApp ] # Steps are a linear sequence of operations that make up a job.
  ```

- variables：定义YAML中用到的变量



checkout 无法指定commit id，只能在想要用的commit id的基础上单拉一个分支