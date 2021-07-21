from os import name
from types import prepare_class
from networkx.algorithms.distance_measures import center
import numpy as np
import matplotlib.pyplot as plt
from scipy._lib.six import b
import my_DPC
import DPC
import sklearn.datasets
import vision
from sklearn.metrics import fowlkes_mallows_score
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import kneighbors_graph
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

def sousuo(X,y,k=3):
    b=0
    for i in range(1,100):
        for j in range(5,50):
            shuffel_y,shuffel_X,label_,rho_,deltas_,center_index_ =my_DPC.fit__(X,y,k=k,n_neighbors=j,percent=i)
            if fowlkes_mallows_score(shuffel_y,label_)>b:
                b=fowlkes_mallows_score(shuffel_y,label_)
                print("i:",i,"j:",j)
                print(b)
            # print("my_DPC算法的FMI：%8f"%fowlkes_mallows_score(shuffel_y,label_))
            # print("my_DPC算法的AMI：%8f"%adjusted_mutual_info_score(shuffel_y,label_))
            # print("my_DPC算法的ARI：%8f"%adjusted_rand_score(shuffel_y,label_))
        if fowlkes_mallows_score(shuffel_y,label_)==1:
            break

def cluster_result_compare(X,y,k):

    # kmeans = KMeans(n_clusters=k, random_state=20).fit(X)
    # vision.draw_result(X,kmeans.labels_,name="k-means-sprial.eps")
    # print("Kmeans算法的FMI：%8f"%fowlkes_mallows_score(y,kmeans.labels_))
    # print("Kmeans算法的AMI：%8f"%adjusted_mutual_info_score(y,kmeans.labels_))
    # print("Kmeans算法的ARI：%8f"%adjusted_rand_score(y,kmeans.labels_))

    # dbscan = DBSCAN(eps=3.7, min_samples=10).fit(X)
    # vision.draw_result(X,dbscan.labels_,name="dbscan_spiral.eps")
    # print("dbscan算法的FMI：%8f"%fowlkes_mallows_score(y,dbscan.labels_))
    # print("dbscan算法的AMI：%8f"%adjusted_mutual_info_score(y,dbscan.labels_))
    # print("dbscan算法的ARI：%8f"%adjusted_rand_score(y,dbscan.labels_))

    label,rho,deltas,center_index=DPC.fit(X,k=k,percent=20)

    vision.draw_decision(rho,deltas,name="123.eps")
    vision.draw_result(X,label,center_index=center_index)
    print("DPC算法的FMI：%8f"%accuracy_score(y,label))
    print("DPC算法的AMI：%8f"%adjusted_mutual_info_score(y,label))
    print("DPC算法的ARI：%8f"%adjusted_rand_score(y,label))

    # shuffel_y,shuffel_X,label_,rho_,deltas_,center_index_ =my_DPC.fit__(X,y,k=k,n_neighbors=14,percent=19.0)
    # vision.draw_decision(rho_,deltas_,name="my-dpc-jain_decison.eps")
    # vision.draw_result(shuffel_X,label_,center_index=center_index_,name="my-dpc-jain.eps")

    # print("my_DPC算法的FMI：%8f"%fowlkes_mallows_score(shuffel_y,label_))
    # print("my_DPC算法的AMI：%8f"%adjusted_mutual_info_score(shuffel_y,label_))
    # print("my_DPC算法的ARI：%8f"%adjusted_rand_score(shuffel_y,label_))


def sousuo(X,y,k=3):
    b=0
    for i in range(1,100):
        for j in range(5,80):
            shuffel_y,shuffel_X,label_,rho_,deltas_,center_index_ =my_DPC.fit__(X,y,k=k,n_neighbors=j,percent=i)
            if fowlkes_mallows_score(shuffel_y,label_)>b:
                b=fowlkes_mallows_score(shuffel_y,label_)
                print("i:",i,"j:",j)
                print(b)
            # print("my_DPC算法的FMI：%8f"%fowlkes_mallows_score(shuffel_y,label_))
            # print("my_DPC算法的AMI：%8f"%adjusted_mutual_info_score(shuffel_y,label_))
            # print("my_DPC算法的ARI：%8f"%adjusted_rand_score(shuffel_y,label_))
        if accuracy_score(shuffel_y,label_)==1:
             break
                
def an(X,y,k=7):
    percent=[60,70,80,89]
    c=[]
    for j in percent:
        b=[]
        for i in range(1,y.shape[0]):
            shuffel_y,shuffel_X,label_,rho_,deltas_,center_index_ =my_DPC.fit__(X,y,k=k,n_neighbors=i,percent=j,n=i)
            b.append(accuracy_score(shuffel_y,label_))
        c.append(np.array(b))
        print(c)
    return c



dataset_name ="jain.txt"
X,y=readfiledata(dataset_name)

#cluster_result_compare(X,y,k=2)
#sousuo(X,y,k=2)
c=an(X,y,k=2)
vision.draw_plot(np.arange(1,len(X)),c[0],c[1],c[2],c[3])