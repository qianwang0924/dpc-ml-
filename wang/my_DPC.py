#
# wangjiaming time:2021/6/1
# 

import numpy as np
import matplotlib.pyplot as plt
import math
import time
from sklearn.manifold import Isomap
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn import datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
import vision

plt.style.use('seaborn-paper')

def readfiledata(path):
    f = open(path)
    data = f.readlines()
    data = [i.split(',') for i in data]
    lables = np.array([j.replace('\n', '') for j in [i[0] for i in data]])
    lables=lables.astype(np.int)-np.ones(lables.shape[0])
    data=[i[1:] for i in data]
    datatable = np.zeros((len(data), len(data[0])))
    for i in range(len(data)):
        datatable[i] = list(map(float, data[i]))
    f.close()
    return datatable, lables




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

    isomap = Isomap(n_components=13,n_neighbors=n_neighbors,path_method="auto")#如果n_neighbors设置的过小，会导致图不可达的现象
    isomap._fit_transform(datas)
    
    #vision.drawG(kneighbors_graph(datas,n_neighbors=n_neighbors))
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
            
            rho[i] = int(np.where(dists[i,:]<dc)[0].shape[0]-1)*np.sum(dists[i,np.where(dists[i,:]<dc)[0]])#找到距离矩阵第i行（每个向量与其他向量之间的距离）小于dc的个数,实现论文中的公式
        else:
            rho[i] = np.where(dists[i,:]<dc)[0].shape[0]-1#软计数，利用高斯公式

    return rho

def get_denity_index(X,v_index,n_neighbors=10):

    Nbrs = NearestNeighbors(n_neighbors=n_neighbors)
    Nbrs.fit(X)
    index=Nbrs.kneighbors(return_distance=False)[v_index]
    #print(index)
    return index



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

def find_center(rho,deltas,type='not auto',k=2):
    center=[]
    if type=='auto':
        rho_threshold = (np.min(rho)+np.max(rho))/2
        daltas_threshold = (np.min(deltas)+np.max(deltas))/2

        N = np.shape(rho)[0]
        for i in range(N):
            if rho[i]>rho_threshold and deltas[i]>daltas_threshold:
                center.append(i)
       # print(center)
    else:
        center=np.argsort(-rho*deltas)[:k]
        #print(center)

    return np.array(center)#返回每个中心点向量index

    
def data_loader(data,target,visions=None,distance_type='isomap',k=2,percent=59.0,n_neighbors=10):
    #混洗样本
    rng=np.random.RandomState(0)
    indices=np.arange(len(data))
    rng.shuffle(indices)
    X=data[indices]
    y=target[indices]
    #可视化原始数据集
    if visions!= None:
        vision.draw_result(X,y)
    

    #密度峰值聚类找到峰值点
    if distance_type == 'isomap':
        dists= getDistanceMatrix_isomap(X,n_neighbors=n_neighbors)
        dc= selrct_dc(dists,percent=percent)#得到合适的dc
        rho = get_denity(dists,dc,method=None)
        deltas,near_neiber= get_deltas(dists,rho)#局部密度
       # vision.draw_decision(rho,deltas)
        center_indices = find_center(rho,deltas,k=k)
        

    else:
        dists= getDistanceMatrix(X)
        dc = selrct_dc(dists,percent=percent)#得到合适的dc
        rho = get_denity(dists,dc,method=None)#局部密度
        deltas,near_neiber= get_deltas(dists,rho)
        #vision.draw_decision(rho,deltas)
        center_indices = find_center(rho,deltas,k=k)


    #unlabeled_indices=np.array(list(set(np.arange(len(y))).difference(set(center_indices).union(set(center_neiber_index)))))

    return X,y,center_indices,rho,dc,deltas,near_neiber



def test_LabelPropagation(data,visions=None,n_neighbors=10,k=8):
    
    X,y,center_indices,rho,dc,deltas=data#dc为密度峰值的密度阈值

    classify = 0
    center_neiber_index=[]
    y_train= np.zeros(np.shape(y)[0])
    for i in center_indices:

        y_train[i] = classify
        indexx = get_denity_index(X,v_index=i,n_neighbors=n_neighbors)
        for j  in indexx:
            y_train[j] = classify
            center_neiber_index.append(j)
        
        classify = classify + 1
    #print(center_indices)
    #print(center_neiber_index)

    unlabeled_indices=np.setdiff1d(np.arange(len(X)),np.union1d(np.array(center_neiber_index),center_indices))
    y_train[unlabeled_indices]=-1
    #print(y_train)
    # print(y_train)
    # print(rho)
    #clf=LabelPropagation(max_iter=100,n_neighbors=n_neighbors,
                            #kernel='knn')
    label_prop_model = LabelSpreading(kernel="knn",n_neighbors=59,max_iter=100)
    label_prop_model.fit(X,y_train)

    #clf.fit(X,y_train)

    predicted_labels = label_prop_model.transduction_[unlabeled_indices]
    y_train[unlabeled_indices] = predicted_labels
    if visions != None:
        vision.draw_result(X,y_train)
    return y_train,rho,deltas,center_indices

def cluster_PD(data):
    X,y,center_indices,rho,dc,deltas,near_neiber=data
    k=np.shape(center_indices)[0]
    if k==0:
        print("can  not find center")
        return
    n=np.shape(rho)[0]
    labs = -1*np.ones(n).astype(int)
    for i,center in enumerate(center_indices):
        labs[center]=i
    index_rho = np.argsort(-rho)
    for i, index in enumerate(index_rho):
        if labs[index]==-1:
            labs[index]=labs[int(near_neiber[index])]
    return labs,rho,deltas,center_indices


def DPC_labelspreading(data,percent):
    X,y,center_indices,rho,dc,deltas,near_neiber=data
    k=np.shape(center_indices)[0]
    if k==0:
        print("can  not find center")
        return
    n=np.shape(rho)[0]
    labs = -1*np.ones(n).astype(int)
    for i,center in enumerate(center_indices):
        labs[center]=i
    index_rho = np.argsort(-rho)
    for i, index in enumerate(index_rho):
        if labs[index]==-1:
            labs[index]=labs[int(near_neiber[index])]
        if i==int((percent/100)*n):
            break

    unindex=index_rho[int((percent/100))*n:]
    label_prop_model = LabelSpreading(kernel="knn",n_neighbors=14,max_iter=100)
    

    label_prop_model.fit(X,labs)
    predicted_labels = label_prop_model.transduction_[unindex]
    labs[unindex] = predicted_labels

    return labs,rho,deltas,center_indices


def fit(X,y,k=7,n_neighbors=32,percent=3.0):
    data = data_loader(X,y,k=k,percent=percent,n_neighbors=n_neighbors)
    y_train,rho,deltas,center_indices =test_LabelPropagation(data,n_neighbors=n_neighbors)
    return data[1],data[0],y_train,rho,deltas,center_indices

def fit_(X,y,k=7,n_neighbors=32,percent=3.0):
    data = data_loader(X,y,k=k,percent=percent,n_neighbors=n_neighbors)
    y_pre,rho,deltas,center_indices = cluster_PD(data)
    return data[1],data[0],y_pre,rho,deltas,center_indices

def fit__(X,y,k=7,n_neighbors=32,percent=3.0,n=30):
    data = data_loader(X,y,k=k,percent=percent,n_neighbors=n_neighbors)
    y_pre,rho,deltas,center_indices=DPC_labelspreading(data,percent=percent)
    return data[1],data[0],y_pre,rho,deltas,center_indices


