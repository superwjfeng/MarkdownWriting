# Introduction

## *UML*

## *软件开发模式*



## *Git*

SVN是集中式版本管理系统，版本库是几种放在中央服务器的，而工作的时候要先从中央服务器拉取最新版本，然后工作完成后再push到中央服务器。集中式版本控制系统必须要联网才能工作，对网络带宽要求较高

Git是分布式版本控制系统，没有中央服务器，每个人本地就有一个完整的版本库。

Git 将顶级目录中的文件和文件夹作为集合，并通过一系列快照 snapshot 来管理其历史记录

每一个状态会指向它之前的状态

git使用有向无环图 directed acyclic graph 来建模历史



Git 中对象根据内容地址寻址，在储存数据时，所有的对象都会基于它们的SHA-1 哈希值进行寻址



staging area 暂存区



master 是主要分支，它是一个对于SHA-1的map，即 `map<string, string>`



vimdiff ?



origin一般用作本地对remote repository的名称