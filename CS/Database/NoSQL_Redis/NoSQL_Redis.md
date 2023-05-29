---
Title: NoSQL_Redis
Author: Weijian Feng 封伟健
---
# 数据类型

## *常用数据类型*

## *数据结构*

# Redis 结构源码

# 持久化

## *AOF*

Append Only File

## *ROB快照*

Redis Database Backup Snapshot

# 缓存 Cache

## *缓存失效*

### 缓存雪崩

Cache Avalanche

### 缓存击穿

Cache Breakdown

### 缓存穿透

Cache Penetration

## *数据库与缓存的一致性*

# 分布式

## *主从复制*

### 建立链接、协商同步

使用 `replicaof` 来形成主从关系：比如说有服务器A和服务器B，我们想要让服务器B变成服务器A的从服务器，那么可以在从服务器B上执行下面的命令

```
# 服务器 B 执行这条命令
replicaof <服务器 A 的 IP 地址> <服务器 A 的 Redis 端口号>
```



### 主服务器同步数据给从服务器

### 主服务器发送新写操作命令给从服务器

## *哨兵机制*

## *集群*