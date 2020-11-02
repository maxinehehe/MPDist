# -*- coding:utf-8 -*-
from __future__ import absolute_import

import stumpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from numpy import fft
import fastdtw
from scipy.spatial.distance import euclidean
# from numpy.fft import fft


'''
根据 论文 《Matrix Profile XII: MPdist: A Novel Time Series Distance Measure to Allow Data Mining in More Challenging Scenarios》
实现 主要分两大板块 
1、实现 同等长度 两个时间序列的比较  MPdistEqualLength(ts1, ts2, L)
2、query-by-content Speeding up MPdist Search 
                    MPdistQueryByContent( Q, T, L):
'''
class MPdist(object):
    def __init__(self):
        pass

    def sqrtxyz(self, x, y, z):
        # 给定三个数 求 sqrt(x^2+y^2+z^2)
        # 遇到的错误 忘了 self, 导致
        # TypeError: ('sqrtxyz() takes 3 positional arguments but 4 were given', 'occurred at index 0')
        return np.sqrt(x*x + y*y + z*z)

    def zNormalize(self, ts):
        """
        Returns a z-normalized version of a time series.
        Parameters
        ----------
        ts: Time series to be normalized
        """

        ts -= np.mean(ts)
        std = np.std(ts)

        if std == 0:
            raise ValueError("The Standard Deviation cannot be zero")
        else:
            ts /= std

        return ts

    def movmeanstd(self, ts, m):
        """
        Calculate the mean and standard deviation within a moving window passing across a time series.
        Parameters
        ----------
        ts: Time series to evaluate.
        m: Width of the moving window.
        """
        if m <= 1:
            raise ValueError("Query length must be longer than one")

        ts = ts.astype("float")
        # Add zero to the beginning of the cumsum of ts
        s = np.insert(np.cumsum(ts), 0, 0)
        # Add zero to the beginning of the cumsum of ts ** 2
        sSq = np.insert(np.cumsum(ts ** 2), 0, 0)
        segSum = s[m:] - s[:-m]
        segSumSq = sSq[m:] - sSq[:-m]

        movmean = segSum / m
        movstd = np.sqrt(segSumSq / m - (segSum / m) ** 2)

        return [movmean, movstd]

    def movstd(self, ts, m):
        """
        Calculate the standard deviation within a moving window passing across a time series.
        Parameters
        ----------
        ts: Time series to evaluate.
        m: Width of the moving window.
        """
        if m <= 1:
            raise ValueError("Query length must be longer than one")

        ts = ts.astype("float")
        # Add zero to the beginning of the cumsum of ts
        s = np.insert(np.cumsum(ts), 0, 0)
        # Add zero to the beginning of the cumsum of ts ** 2
        sSq = np.insert(np.cumsum(ts ** 2), 0, 0)
        segSum = s[m:] - s[:-m]
        segSumSq = sSq[m:] - sSq[:-m]

        return np.sqrt(segSumSq / m - (segSum / m) ** 2)

    def slidingDotProduct(self, query, ts):
        """
        Calculate the dot product between a query and all subsequences of length(query) in the timeseries ts. Note that we use Numpy's rfft method instead of fft.
        Parameters
        ----------
        query: Specific time series query to evaluate.
        ts: Time series to calculate the query's sliding dot product against.
        """

        m = len(query)
        n = len(ts)

        # If length is odd, zero-pad time time series
        ts_add = 0
        if n % 2 == 1:
            ts = np.insert(ts, 0, 0)
            ts_add = 1

        q_add = 0
        # If length is odd, zero-pad query
        if m % 2 == 1:
            query = np.insert(query, 0, 0)
            q_add = 1

        # This reverses the array
        query = query[::-1]

        query = np.pad(query, (0, n - m + ts_add - q_add), 'constant')

        # Determine trim length for dot product. Note that zero-padding of the query has no effect on array length, which is solely determined by the longest vector
        trim = m - 1 + ts_add

        dot_product = fft.irfft(fft.rfft(ts) * fft.rfft(query))

        # Note that we only care about the dot product results from index m-1 onwards, as the first few values aren't true dot products (due to the way the FFT works for dot products)
        return dot_product[trim:]

    def mass(self, query, ts):
        """
        Calculates Mueen's ultra-fast Algorithm for Similarity Search (MASS): a Euclidian distance similarity search algorithm.
        Note that we are returning the square of MASS.
        Again Note that we have modified that returning sqrt of MASS.
        Parameters
        ----------
        :query: Time series snippet to evaluate. Note that the query does not have to be a subset of ts.
        :ts: Time series to compare against query.

        Input: A query Q, and a user provided time series T
        Output: A distance profile D of the query Q
        """
        # 标准化
        # query_normalized = zNormalize(np.copy(query))
        m = len(query)
        q_mean = np.mean(query)
        q_std = np.std(query)
        mean, std = self.movmeanstd(ts, m)
        dot = self.slidingDotProduct(query, ts)

        # res = np.sqrt(2*m*(1-(dot-m*mean*q_mean)/(m*std*q_std)))
        res = 2 * m * (1 - (dot - m * mean * q_mean) / (m * std * q_std))

        return res

    def MPdistEqualLength(self, ts1, ts2, L):
        '''
        此方法用于实现 两个 长度相等的时间序列 求他们的 MPdist 距离 ， 需要的参数 ， ts1, ts2, L
        :param ts1: 时间序列 ts1
        :param ts2: 时间序列 ts2
        :param L:  子序列的长度（即窗口的长度）
        :return: 第 k-th 大小的值 这里的 k-th = 2*n*0.05 即 时间序列 ts1 和 ts2 的总长度的 5%
        '''
        lenA = len(ts1)
        lenB = len(ts2)
        # 如果两个时间序列不相等 则提出出错 此函数是处理相等长度的时间序列的比较
        if lenA != lenB:
            warnings.warn("输入的两条时间序列不相等！请使用 MPdistQueryByContent 函数。", UserWarning)

            exit(0)
        # 计算P_AB <ts1, ts2>
        # 这里使用 stumpy 计算
        mpDataAB = stumpy.stump(ts1, L, ts2, ignore_trivial=False)
        mpAB = mpDataAB[:, 0]   # 获取 P_AB 的矩阵轮廓
        mpDataBA = stumpy.stump(ts2, L, ts1, ignore_trivial=False)
        mpBA = mpDataBA[:, 0]   # 获取 P_BA 的矩阵轮廓
        # 进行合并 mpAB 和 mpBA 得到 P_ABBA
        P_ABBA = np.concatenate((mpAB, mpBA))
        # 对 P_ABBA 进行排序
        P_ABBA_sorted = np.sort(P_ABBA)
        lenAB = lenA + lenB
        # 总长度 的 %5 几位 k-th 我们所求的 第 k 个 最小的值
        k = int(np.round(lenAB * 0.05))
        print("K: ", k)
        len_P_ABBA = len(P_ABBA)
        # 判断 k 的长度  如果 P_ABBA 的长度 不大于 k 则直接返回最大值 （欧几里德）
        if len_P_ABBA > k:
            # print(P_ABBA_sorted)
            return (P_ABBA_sorted[k - 1])
        else:
            return np.max(P_ABBA_sorted)


    def MPdistQueryByContent(self, Q, T, L):
        """
        :param Q: 查询子序列
        :param T: 时间序列 T
        :param L: 窗口大小
        :return: a distance vector(MPdist_vect) that contains the MPdist between Q and T_i:i+n ,
                for all i in the range 1 to m-n+1
        这个函数我们用于 计算 一个 m 长 序列 Q 和 n 长 序列 T ，产生一个 MPdist距离向量
        可用于比较两个不同长度的序列的MPdist 结果会产生一个 距离向量
        具体做法 参见 论文《Matrix Profile XII: MPdist: A Novel Time Series Distance Measure to Allow Data Mining in More Challenging Scenarios》
                中 C.Speeding up MPdist Search 小节
        """
        # 获取 序列 Q 和 T 的长度
        lenQ = len(Q)
        lenT = len(T)
        if(lenQ > lenT):
            warnings.warn("输入 第一个参数 大于 第二参数！进行替换 两个参数 Q T 操作。", UserWarning)
            tmp = Q
            Q = T
            T = tmp
            lenQ = len(Q)
            lenT = len(T)

        n = lenQ    # 窗口的大小
        mpdistVect = []   # 保存距离向量
        P_AB = []
        P_BA = []
        P_ABBA = []
        k = int(np.round(2*lenQ*0.05))    # 四舍五入并取整
        distProfile = np.zeros(0)
        subSeqNum = lenQ - L + 1     # 矩阵 维数
        # 下面开始进行 取滑窗 step 为 1 窗口的大小为 n(lenQ,即 Q 的长度)
        for ti in range(0, lenT-lenQ+1):
            for qi in range(lenQ-L+1):
                # 使用MASS算法开始计算距离轮廓
                if qi == 0:
                    # tmpDistProfile = np.zeros(0)   # 初始化
                    tmpDistProfile = self.mass(Q[qi:qi+L], T[ti:ti+lenQ])  # T[ti:ti+lenQ]
                    distProfile = np.vstack((tmpDistProfile, ))
                else:
                    tmpDistProfile = self.mass(Q[qi:qi+L], T[ti:ti+lenQ])   # T[ti:ti+lenQ]
                    distProfile = np.vstack((distProfile, tmpDistProfile ))
            # 一次循环结束 获取 P_AB 和 P_BA 以及 P_ABBA

            if len(P_AB) == 0 or len(P_BA) == 0:   # 第一次处理开始
                for i in range(subSeqNum):
                    # 下面 计算矩阵轮廓 每一行 和 每一列的 最小值
                    P_AB.append(np.min(distProfile[i, :]))    # 获取每一行的最小值
                    P_BA.append(np.min(distProfile[:, i]))  # 获取每一列的最小值
                # P_ABBA = np.concatenate((P_AB, P_BA))
                # P_ABBA_sorted = np.sort(P_ABBA)
                # if len(P_ABBA_sorted) > k:
                #     # print(P_ABBA_sorted)
                #     mpdistVect.append(P_ABBA_sorted[k - 1])
                # else:
                #     mpdistVect.append(np.max(P_ABBA_sorted))
            else:
                # 表示不是第一次 后面 对于每一列 P_BA 则可以直接 排出第一个值 放入一个新的值
                P_BA.pop(0)  # 排出第一位
                P_BA.append(np.min(distProfile[:, -1]))
                # 而对于 P_AB 则需要 O(subSeqNum)
                P_AB = []
                for i in range(subSeqNum):   # 搜索P_AB的最小值
                    P_AB.append(np.min(distProfile[i, :]))
            P_ABBA = np.concatenate((P_AB, P_BA))
            P_ABBA_sorted = np.sort(P_ABBA)
            if len(P_ABBA_sorted) > k:
                # print(P_ABBA_sorted)
                mpdistVect.append(P_ABBA_sorted[k - 1])
            else:
                mpdistVect.append(np.max(P_ABBA_sorted))


        return mpdistVect    # 返回搜索到的 MPdist_vec

    def distDTW(self, Q, T, L):
        distVec = []
        lenQ = len(Q)
        lenT = len(T)
        for ti in range(0, lenT-lenQ+1, 165):
            res = fastdtw.dtw(Q, T[ti:ti+L], euclidean)[0]
            print(" idx : ", ti, " curr dist: ", res)
            distVec.append(res)
        return distVec





if __name__ == '__main__':

    # 测试
    mpdist = MPdist()
    data = np.array([0.21105276, 0.21105276, 0.21105276, 0.21105276, 0.21105276, 0.21105276, 0.38618448, 0.44013295, 0.44013295, 0.44013295, 0.44013295, 0.44013295, 0.44013295, 0.44013402, 0.44013402, 0.44013402, 0.44013402, 0.44013402, 0.44013402, 0.50303584, 0.50303584, 0.50303584, 0.50303584, 0.50303584, 0.50303584, 0.5455245, 0.5455245, 0.5455245, 0.5455245, 0.5455245, 0.5455245, 0.62646085, 0.62646085, 0.62646085, 0.62646085, 0.62646085, 0.64302766, 0.64302766, 0.64302766, 0.64302766, 0.64302766, 0.64302766, 0.70470744, 0.70470744, 0.7693379, 0.8693771, 0.87758744, 1.0484976, 1.0709115, 1.1456468, 1.1456468, 1.1456468, 1.1456468, 1.1456468, 1.245265, 1.245265, 1.245265, 1.3100065, 1.3100065, 1.3100065])
    data2 = np.random.rand(20)
    data3 = np.random.rand(20)

    # cane_100 = np.loadtxt(r"E:\奇妙文档包\毕业论文开题\相关论文\活动识别语义分割\MPdist资料\MPdistVect_MPdist\MPdistNew\Cane_100_2345.txt", delimiter="\n")
    # Q = cane_100[:200]
    # T = cane_100[:500]
    # # print(mpdist.MPdistEqualLength(T, Q, 100))
    # res = mpdist.MPdistQueryByContent(T, Q, 100)
    # print(res)
    # print(len(res))

    # slw_data = pd.read_csv(r"E:\奇妙文档包\毕业论文开题\数据集\selfTest\SIT_CHU_WALK_STD_WALK.csv", sep=",", engine='python')
    slw_data = pd.read_csv(r"E:\奇妙文档包\毕业论文开题\数据集\MobiAct_Dataset_v2.0.rarMobiAct_Dataset_v2.0\MobiAct_Dataset_v2.0\Annotated Data\WAL\WAL_2_1_annotated.csv", sep=",", engine='python')
    slw_mod_data = slw_data.iloc[:, 2:5][:]

    slw_mod_data['val_sqrt'] = slw_mod_data.apply(lambda row: mpdist.sqrtxyz(row['acc_x'], row['acc_y'], row['acc_z']), axis=1)

    slw_data2 = pd.read_csv(
        r"E:\奇妙文档包\毕业论文开题\数据集\MobiAct_Dataset_v2.0.rarMobiAct_Dataset_v2.0\MobiAct_Dataset_v2.0\Annotated Data\WAL\WAL_1_1_annotated.csv",
        sep=",", engine='python')
    slw_mod_data2 = slw_data2.iloc[:, 2:5][:5000]
    slw_mod_data2['val_sqrt'] = slw_mod_data2.apply(lambda row: mpdist.sqrtxyz(row['acc_x'], row['acc_y'], row['acc_z']), axis=1)

    # print(slw_mod_data.head())
    # Q = np.array(slw_mod_data.iloc[:, 3][1500:1500+87])
    # print("Q: ", len(Q))
    # T = np.array(slw_mod_data.iloc[:, 3])
    # print("T length: ", len(T))
    # res = mpdist.MPdistQueryByContent(Q, T, 87)
    # print("结果：\n", res)
    # res1 = mpdist.distDTW(Q, T, 220)

    # res = mpdist.MPdistEqualLength(slw_mod_data.iloc[:, 3][100:3000], slw_mod_data2.iloc[:, 3][100:3000], 87)
    # WAL-WAL 2.58614769525459/1.52 /3-2.3 / 6-1.89
    # WAL-STD 6.58614769525459/5.18 /3-4.79 /6- 4.81
    # WAL-SIT 3.58614769525459/4.79 /3-2.83 /6-4.04
    # res = fastdtw.dtw(slw_mod_data.iloc[:, 3][100:3000], slw_mod_data2.iloc[:, 3][100:3000], euclidean)
    # WAL-WAL 2969
    # print(res)


    # resDist = mpdist.MPdistEqualLength(slw_mod_data.iloc[:, 3][0:870], slw_mod_data.iloc[:, 3][1200:1200+870], 87)
    # print(resDist)
    #
    # plt.figure(figsize=(20, 10))
    # plt.plot(np.arange(len(T[:])), T[:])
    # plt.title('stupy.fluss->CAC')
    # plt.show()
    #
    # print(len(res))
    # plt.figure(figsize=(20, 10))
    # plt.plot(np.arange(len(res)), res)
    # plt.title('stupy.fluss->CAC')
    # plt.show()
    #
    # plt.figure(figsize=(20, 10))
    # plt.plot(np.arange(len(T[:5000])), T[:5000])
    # plt.title('stupy.fluss->CAC')
    # plt.show()


    # import fastdtw
    # from scipy.spatial.distance import euclidean
    # print("fastDtw: ", fastdtw.dtw(slw_mod_data.iloc[:, 3][0:870], slw_mod_data.iloc[:, 3][2000:2000+870], euclidean)[0])
    # print("mpdist: ", mpdist.MPdistEqualLength(slw_mod_data.iloc[:, 3][0:870], slw_mod_data.iloc[:, 3][2000:2000+870], 87))
    # 1347 不相关的
    # 6.8相关的   220 ： 15.17

    # 相关的     fastdtw :366   mpdist 6.14   220: 14





