# 随机过程 Stochastic Process

# 概率图

## *概率图模型引入*

### 概率论公式

* Sum rule
  $$
  p(x_1)=\int{p(x_1,x_2)dx_2}
  $$

* Product rule
  $$
  p(x_1,x_2)=p(x_1|x_2)p(x_2)=p(x_2|x_1)p(x_21)
  $$

* Chain rule

$$
p(x_1,x_2,\dots,x_p)=\prod\limits_{i=1}^{p}{\left[p(x_i)|\bigcap\limits_{j=1}^{i-1}{p)x_j}\right]}
$$

* Bayersian rule
  $$
  p(H|E)=\frac{p(H)\cdot p(E|H)}{\int{p(H)p(E|H)dH}}
  $$

可以看出，计算特别复杂

* 概率图模型 $G=(V,E)$ 采用图的特点表示上述的条件独立性假设，节点表示随机变量，边表示条件概率
* 通过概率图可以
  * provides a particular factorization for the joint distribution $p(x_1,x_2,\dots,x_V)$
  * provides a set of conditional independencies, inferred by a routine D-separation

### 概率图模型总结

* 体系划分

  * 表示 Representation

    * 一般为离散
      * 有向图 Directed Graph：Bayersian Network
      * 无向图 Undirected Graph：Markov Network
    * 连续：高斯图
      * Gaussian Bayersian Network
      * Gaussian Markov Network

    * 推断 Inference
      * 精确推断
      * 近似推断
        * 确定性近似 变分推断
        * 随机近似 MCMC

  * 学习 Learning
    * 参数学习
      * 完备数据
      * 隐变量：E-M 算法
      * 结构学习

* 根据条件独立性对应用模型进行划分

  * 单一的条件独立性假设：朴素贝叶斯
  * 混合的条件独立：高斯混合模型
  * 与时间相关的条件以来
    * Markov链
    * 高斯过程（无限维高斯分布）
  * 连续：高斯贝叶斯网络
  * 组合上面的分类：GMM与时序结合的动态模型
    * 离散HMM
    * 线性动态系统 LDS（Kalman滤波）
    * 粒子滤波（非高斯、非线性）

## *有向图：贝叶斯网络*

### 图结构与条件独立性的关系

* 一字型
* $\wedge$ 形
* $\vee$ 形

### D划分 D-Separation

## *无向图：马尔科夫网络/马尔可夫随机场*

## *两种图之间的转换*

## *推断 Inference*

推断的主要目的是求各种概率分布，包括边缘概率，条件概率，以及使用 MAP 来求得参数。推断可以如下划分

* 精确推断
  * Variable Elimination(VE)
  * Belief Propagation(BP, Sum-Product Algo)，从 VE 发展而来
  * Junction Tree，上面两种在树结构上应用，Junction Tree 在图结构上应用
* 近似推断
  * Loop Belief Propagation（针对有环图）
  * Mente Carlo Interference：例如 Importance Sampling，MCMC
  * Variational Inference

### 推断 -- 变量消除 Variable Elimination

### 推断 -- 信念传播 Blief Propagation

### Max-Product 算法

# 变分推断 Variational Inference

# 高斯网络