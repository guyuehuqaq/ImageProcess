# 1、ORB（Oriented FAST and Rotated BRIEF）
ORB（Oriented FAST and Rotated BRIEF） 是一种结合了
FAST 特征点检测 和 BRIEF 特征描述子 的改进算法，兼顾了速度与鲁棒性，并且不受专利限制。
# 2、算法流程
## 1. 特征点检测: FAST角点检测
- ORB 首先用 FAST (Features from Accelerated Segment Test) 算法检测角点。
- 原理：在候选像素点周围画一个半径为 3 的圆，比较 16 个像素值与中心点像素值差异， 
如果有连续 N 个点比中心点亮很多或暗很多，就判定为角点。
- ORB 改进:
    - 使用 多尺度金字塔，在不同尺度图像上检测 FAST 角点 → 保证尺度不变性。
    - 通过 Harris Corner Response 对 FAST 检测到的点打分，选取最稳定的前 N 个关键点。

## 2. 关键点方向分配
- FAST 检测的点本身没有方向。ORB 引入一种 强度质心法 (Intensity Centroid) 来估计方向:
    - 计算关键点邻域的灰度矩:
      
       `mpq = Σx Σy (x^p * y^q * I(x,y))`
    - 质心位置为:

       `C = ( m10 / m00 , m01 / m00 )`
  
    - 方向角:
        
        `θ = arctan( m01 / m10 )`
- 这样每个关键点就有了一个旋转角度，用于后续构建旋转不变的描述子

## 3. 特征描述子: 改进的BRIEF
- BRIEF（Binary Robust Independent Elementary Features）是基于强度差的二进制描述子。
- 在关键点邻域随机采样点对 (p, q),对每个点对比较灰度值：

        `τ(p, q) = 1,  如果 I(p) < I(q)`
        `τ(p, q) = 0,  如果 I(p) ≥ I(q)`
- ORB 改进：
    - 旋转不变性：对 BRIEF 的采样点对，应用关键点方向角 θ 做旋转 → 得到 rBRIEF (rotated BRIEF)。
    - 学习最佳点对：通过训练，选择信息量最大、相关性最小的点对，保证描述子更稳定。
    - 默认长度 256 bits（32 字节），比 SIFT/SURF 描述子更紧凑。
## 4. 特征匹配
- ORB 描述子是二进制的 → 用 汉明距离 (Hamming Distance) 比较。
- 常用最近邻 + 最近邻比率（NNDR）做匹配。
- 可以用 FLANN 或者暴力匹配（BruteForce-Hamming）。


