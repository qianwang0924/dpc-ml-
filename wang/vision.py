
import networkx as nx
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')


def drawG(g):
    fig, ax = plt.subplots()
    plt.figure(1)
    G = nx.Graph(g)
    nx.draw(G,node_size = 10,width=0.5,pos = nx.spring_layout(G),alpha=0.85)
    fig.savefig("g.eps",dpi=600,format='eps')
    plt.show()


def draw_result(X,y,name=None,center_index=None):
    fig, ax = plt.subplots()
    plt.figure(1)
    plt.scatter(X[:,0],X[:,1],c=y,s=8)
    try:
        if center_index.all()!=None:
            plt.scatter(X[center_index,0], X[center_index,1], color='', marker='o', edgecolors='r', s=23)
    except:
        pass
    if name != None:
        fig.savefig(name,dpi=600,format='eps')
    plt.show()


def draw_decision(X,y,name=None):
    fig, ax = plt.subplots()
    plt.figure(1)
    plt.scatter(X,y,s=8)
    if name != None:
        fig.savefig(name,dpi=600,format='eps')
    plt.show()

def draw_plot(x,y1,y2,y3,y4):
    fig, ax = plt.subplots()
    plt.figure(1)
    plt.plot(x,y1,'o-',color = 'r',label="percent=10",linewidth=1,markersize=2)#s-:方形
    plt.plot(x,y2,'o-',color = 'g',label="percent=20",linewidth=1,markersize=2)#o-:圆形
    plt.plot(x,y3,'o-',color = 'b',label="percent=30",linewidth=1,markersize=2)#o-:圆形
    plt.plot(x,y4,'o-',color = 'y',label="percent=40",linewidth=1,markersize=2)#o-:圆形
    plt.xlabel("k")#横坐标名字
    plt.ylabel("accuracy")#纵坐标名字
    plt.legend(loc = "best")#图例
    fig.savefig("go.eps",dpi=600,format='eps')
    plt.show()
