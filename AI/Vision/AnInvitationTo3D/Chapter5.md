# Reconstruction from Two Calibrated Views

## *Epipolar geometry*

## *Basic reconstruction algorithms*

### 八点算法

* 有 $E=\hat{T}R=\left[\begin{array}{c}e_{11}&e_{12}&e_{13}\\e_{21}&e_{22}&e_{23}\\e_{31}&e_{32}&e_{33}\end{array}\right]\in\mathbb{R}^{3\times 3}$。Stack $E$ into a vector $E^s=\left[e_{11},e_{12},e_{13},e_{21},e_{22},e_{23},e_{31},e_{32},e_{33}\right]^T\in\mathbb{R^9}$
* $x_1=\left[x_1,y_1,z_1\right]^T\in\mathbb{R^3},\ x_2=\left[x_2,y_2,z_2\right]^T\in\mathbb{R^3}\Rightarrow a=x_1\otimes x_2=\left[\begin{array}{c}x_1x_2,x_1y_2,x_1z_2,y_1x_2,y_1y_2,z_1x_2,z_1y_2,z_1z_2\end{array}\right]^T$
* 给一组对应点 $(x_1^j,x_2^j),\ j=1,2,...,n$，定义每组对应点的克罗内克积组成的为 $\chi\triangleq \left[\begin{array}{c}a^1,a^2,...,a^n\end{array}\right]^T\in\mathbb{n\times 9}$。八点算法的意义只是指最少需要八个点来求本质矩阵，实际的算法并不是只取八个点
* 根据极几何约束 $x_2^TEx_1=0=a^TE^s\rightarrow\chi E^s=0$
* SVD求解齐次 $E$ 最小二乘解 $\hat{E}$
* 由于 $rank(E)=2$，但求出来的 $E$ 是满秩的，所以Project onto the essential space of rank 2，对 $\hat{E}$ 进行SVD分解，计算 $U\left[\begin{array}{c}\sigma_1&0&0\\0&\sigma_2&0\\0&0&0\end{array}\right]T^T$

### 

## *Planar scenes and homography 平面场景及单应矩阵*