# Kalman Filter

* Kalman Filter: An Optimal Recursive Data Processing Algorithm，一种最优化递归数据处理算法。
* KF的广泛应用是因为真实世界中存在大量的不确定性
  * 不存在完美的数学模型
  * 系统的扰动不可控，也很难建模
  * 测量传感器存在误差

## *递归算法 Recursive Processing*

### 测量例子

* 不同的人用同一把尺子来测量同一个硬币，得到的测量结果为 $z_k=\left\{z_1,z_2,\cdots,z_n\right\}$
* 为了估计真实数据，就要取测量的期望或平均值：$\widehat{x}_k=\frac{1}{k}\left(z_1+z_2+\cdots+z_k\right)=\frac{1}{k}\left(z_1+z_2+\cdots+z_{k-1}\right)+\frac{1}{k}z_k=\frac{1}{k}(k-1)\underbrace{\frac{1}{k-1}(z_1+z_2+\cdots+z_{k-1})}_{\widehat{x}_{k-1}}+\frac{1}{k}z_k=\frac{k-1}{k}\widehat{x}_{k-1}+\frac{1}{k}z_k=\widehat{x}_{k-1}-\frac{1}{k}\widehat{x}_{k-1}+\frac{1}{k}z_k=\widehat{x}_{k-1}+\frac{1}{k}(z_k-\widehat{x}_{k-1})\Rightarrow\widehat{x}_k=\widehat{x}_{k-1}+\frac{1}{k}(z_k-\widehat{x}_{k-1})$
* 分析上式可知：$k\uparrow,\lim\limits_{k\rightarrow \infty}{\frac{1}{k}}=0$，此时有 $\widehat{x}_k\rightarrow\widehat{x}_{k-1}$，即随着测量次数 $k$ 的增加，测量值不再重要，这意思着当拥有大量数据时，对数据的结果很有把握；相反，当 $k$ 值较小时，$\frac{1}{k}$ 较大，$z_k$ 作用就比较大
* 另 $\frac{1}{k}=K_k$，并称为 Kalman Gain 卡尔曼增益/因数，原式变成 $\widehat{x}_k=\widehat{x}_{k-1}+K_k(z_k-\widehat{x}_{k-1})$

### 引出递归算法

* $\widehat{x}_k=\widehat{x}_{k-1}+K_k(z_k-\widehat{x}_{k-1})$ 意味着当前估计值与上次估计值之间的递归关系
* 引入估计误差 $e_{EST}$ 和测量误差 $e_{MSA}$，有Kalman Gain的关系 $K_k=\frac{e_{EST,k-1}}{e_{EST,k-1}+e_{MEA,k}}$（之后会推导）
* 在 $k$ 时刻，当 $e_{EST,k-1}\gg e_{MEA,k},\Rightarrow K_k\rightarrow 1$，有 $\widehat{x}_k=\widehat{x}_{k-1}+z_k-\widehat{x}_{k-1}=z_k$。即当估计误差远大于测量误差时，此时估计值就趋近于测量值，因为估计误差大，而测量准确，因此要信任测量值；相反当 $e_{EST,k-1}\ll e_{MEA,k},\Rightarrow K_k\rightarrow 0$，有 $\widehat{x}_k=\widehat{x}_{k-1}$，即测量误差很大时，要相信估计值

### 算法步骤

* 计算Kalman Gain $K_k=\frac{e_{EST,k-1}}{e_{EST,k-1}+e_{MEA,k}}$
* 计算估算值 $\widehat{x}_k=\widehat{x}_{k-1}+K_k(z_k-\widehat{x}_{k-1})$
* 更新 $e_{EST,k}=(1-K_k)e_{EST,k-1}$（后面会推导）

## *数学工具*

### 数据融合 Data Fusion

### 协方差矩阵 Corvariance Matrix

### 状态空间方程

### 观测器

## **

## **

## **

## **