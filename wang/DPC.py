

import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.core.defchararray import center, index
from sklearn.manifold import Isomap
from sklearn.metrics import cluster
from sklearn.semi_supervised import LabelPropagation
from sklearn import datasets
from sklearn.neighbors import kneighbors_graph
import vision



def dist(a, b):
	return math.sqrt(np.power(a - b, 2).sum())

def getDistanceMatrix(datas):
    #计算欧氏距离

    N,D= np.shape(datas)
    dists = np.zeros([N,N])

    for i in range(N):
        for j in range(N):
            vi = datas[i,:]
            vj = datas[j,:]
            dists[i,j]=dist(vi,vj)
    return dists

def getDistanceMatrix_isomap(datas,n_neighbors):
    #计算测地距离
    N = np.shape(datas)[0]

    isomap = Isomap(n_components=2,n_neighbors=n_neighbors,path_method="auto")#如果n_neighbors设置的过小，会导致图不可达的现象
    isomap._fit_transform(datas)
    geo_distance_metrix = isomap.dist_matrix_
    
    return geo_distance_metrix

def selrct_dc(dists,percent):
    ##适配数据集寻找一个合适的阈值dc，实现密度的选择##
    N = np.shape(dists)[0]
    re = np.reshape(dists,N*N)#将距离矩阵展开，以便于排序
    position =int(N*(N-1)*percent/100)#找到距离在2%的位置
    dc = np.sort(re)[position+N]#找到距离
    return dc

def get_denity(dists,dc,method=None):
    N=np.shape(dists)[0]
    rho = np.zeros(N)#创建每个点的局部密度
    for i in range(N):
        if method == None:
            rho[i] = np.where(dists[i,:]<dc)[0].shape[0]-1#找到距离矩阵第i行（每个向量与其他向量之间的距离）小于dc的个数
        else:
            rho[i] = np.sum(np.exp(-(dists[i,:]/dc)**2))-1#软计数，利用高斯公式
    return rho


def get_denity_index(dists,dc,v_index):
    index=np.where(dists[v_index,:]<dc)[0]
    return np.array(index)


def get_deltas(dists,rho):
    #计算每个向量密度，比该向量大且距离最近的向量
    N = np.shape(dists)[0]
    deltas = np.zeros(N)
    near_neiber = np.zeros(N)
    sort_rho = np.argsort(-rho)

    for i,index in enumerate(sort_rho):
        if i == 0:
            continue
        index_higher_rho = sort_rho[:i]
        #选择距离最近的向量，形成距离列表。同时索引代表每个向量的顺序
        deltas[index] = np.min(dists[index,index_higher_rho])
        index_nn = np.argmin(dists[index,index_higher_rho])
        #保存距离最近的向量的坐标，即index
        near_neiber[index] = index_higher_rho[index_nn].astype(int)
    deltas[sort_rho[0]] = np.max(deltas)
    #返回距离列表，和最近向量的index
    return deltas,near_neiber

def find_center(rho,deltas,type='not auto',k=None):
    center=[]
    if type=='auto':
        rho_threshold = (np.min(rho)+np.max(rho))/2
        daltas_threshold = (np.min(deltas)+np.max(deltas))/2
        N = np.shape(rho)[0]
        for i in range(N):
            if rho[i]>rho_threshold and deltas[i]>daltas_threshold:
                center.append(i)
    else:
        center=np.argsort(-rho*deltas)[:k]

    return np.array(center)#返回每个中心点向量index

def cluster_PD(rho,centers,nearest_neiber):
    k=np.shape(centers)[0]
    if k==0:
        print("can  not find center")
        return
    n=np.shape(rho)[0]
    labs = -1*np.ones(n).astype(int)
    for i,center in enumerate(centers):
        labs[center]=i
    index_rho = np.argsort(-rho)
    for i, index in enumerate(index_rho):
        if labs[index]==-1:
            labs[index]=labs[int(nearest_neiber[index])]
    return labs

def fit(X,k=7,percent=3.0):
    dists= getDistanceMatrix(X)
    dc = selrct_dc(dists,percent)#得到合适的dc
    rho = get_denity(dists,dc,method=None)#局部密度
    deltas,near_neiber= get_deltas(dists,rho)
    center_indices = find_center(rho,deltas,k=k)
    #vision.draw_decision(rho,deltas)
    y=cluster_PD(rho,center_indices,near_neiber)
    #vision.draw_result(X,y)
    return y,rho,deltas,center_indices

