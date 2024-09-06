# Azure Pipelines

[Azure Pipelines 文档 - Azure DevOps | Microsoft Learn](https://learn.microsoft.com/zh-cn/azure/devops/pipelines/?view=azure-devops)

> Azure Pipelines 自动生成和测试代码项目。 它支持所有主要语言和项目类型，并结合了[持续集成](https://learn.microsoft.com/zh-cn/azure/devops/pipelines/get-started/what-is-azure-pipelines?view=azure-devops#continuous-integration)、[持续交付](https://learn.microsoft.com/zh-cn/azure/devops/pipelines/get-started/what-is-azure-pipelines?view=azure-devops#continuous-delivery)和[持续测试](https://learn.microsoft.com/zh-cn/azure/devops/pipelines/get-started/what-is-azure-pipelines?view=azure-devops#continuous-testing)，可以生成、测试代码并将其交付到任何目标

## *创建管道*

在repo的根目录中使用名为 `azure-pipelines.yml` 的 YAML 文件来定义管道

YAML各个字段的意义可以参考：[YAML schema reference | Microsoft Learn](https://learn.microsoft.com/en-us/azure/devops/pipelines/yaml-schema/?view=azure-pipelines)

- trigger：当 Git 仓库的`master`分支有新的提交时，该流水线即会被触发

- pool：定义要在哪个代理池上运行作业。该流水线将会在一台 Linux 节点上运行，节点采用的镜像为`ubuntu-latest`

- variables：定义YAML中用到的变量

- resources：定义外部资源，如repositories, pipelines, containers等

  - repositories：用于指定额外的源代码仓库资源。可用于在同一Azure DevOps组织中或者在GitHub等第三方服务中检出其他仓库的代码

    ```yaml
    resources:
      repositories:
      - repository: MyOtherRepo   # 名称用于在yaml文件中引用
        type: git                 # 类型（例如git, github等）
        name: MyProject/MyRepo    # 仓库的全名（例如Azure DevOps项目/仓库或GitHub用户/仓库）
        ref: refs/heads/main      # 可选，指定要检出的分支或标签
    ```

  - pipelines：允许将其他Azure DevOps pipeline作为资源。这可能用于下载由其他pipelines产生的构建工件，或是触发其他pipeline的执行

    ```yaml
    resources:
      pipelines:
      - pipeline: MyPipeline       # 资源别名
        project: MyProject         # Azure DevOps项目名称
        source: MyOtherPipeline    # 其他pipeline的名称
        branch: main               # 分支
        version: 1234567           # 特定的构建号
    ```

  - containers：如果构建或部署步骤需要运行在特定的容器镜像中，可以预先定义容器资源

    ```yaml
    resources:
      containers:
      - container: my_container     # 容器资源的别名
        image: node:12.13           # 镜像名称，可以来自Docker Hub或私有容器注册表
        options: --hostname my-host # Docker运行参数
    ```

- stages：允许将pipeline分成多个阶段，例如Build, Test, Deploy

  ```yaml
  stages:
  - stage: BuildStage
    displayName: Build and Test # 可选字段，用于在Web界面上显示更友好的阶段名称
    dependsOn: BuildStage # 指定当前阶段是否依赖于前面的某个阶段完成
    variables:
      buildConfiguration: Release
    jobs:
    - job: BuildJob
      # ...
  ```

  dependsOn 如果未指定，则默认依赖于前面定义的所有阶段。可以被设置为显式地依赖某个阶段，或者设置为 `[]` 来表示当前阶段无需等待任何其他阶段

- strategy: 定义部署策略，例如canary, blue-green等

- jobs：job的集合，里面还有很多job

  - steps：job中执行的一步

    ```
    steps: [ task | script | powershell | pwsh | bash | checkout | download | downloadBuild | getPackage | publish | template | reviewApp ] # Steps are a linear sequence of operations that make up a job.
    ```

    checkout 无法指定commit id，只能在想要用的commit id的基础上单拉一个分支


# Jira

https://zhuanlan.zhihu.com/p/686629188

JIRA是澳大利亚的Atlassian公司开发的一款项目管理软件，比如作为仓库自动化工具、管理文档流程、优化费用等等