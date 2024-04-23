import math
import random
import json
import numpy as np
from sklearn.datasets import make_blobs
from Cluster.improved_min_max_kmeans import ImprovedMinMaxKMeans
import matplotlib.pyplot as plt

#Toy dataset
def make_clients(args,n_samples, cmp_cap, com_cap):
    time = [0] * n_samples
    cmp_time = [0] * n_samples
    for i in range(n_samples):
        U = cmp_cap[i][0]
        f = cmp_cap[i][1]
        delta = cmp_cap[i][2]
        D = args.data_num
        cmp_time_c = (D*U*delta)/f

        B = com_cap[i][0]
        # p=com_cap[i][1]
        # g=com_cap[i][2]
        # N0= -104
        if args.model == "logistic":
            M = 100
        else:
            M = 200
        # com_time =M/ (B[0]* math.log2(1+p*g/N0))
        com_time = M / B[0]
        time[i]=cmp_time_c+com_time
        cmp_time[i] = cmp_time_c
    time = np.array(time)
    return time, cmp_time



# X,y = make_blobs(n_samples=100,
#                 random_state=1,
#                 n_features=2,
#                 centers=5)
def cluster(n_samples, centers,args,clients_bds):

    cmp_cap= [[0]*3 for i in range(n_samples)]
    com_cap= [[0]*3 for i in range(n_samples)]
    for i in range(n_samples):
        #U
        cmp_cap[i][0]= random.randint(20,25)
        #frequency
        cmp_cap[i][1] = random.uniform(1000000, 2000000)
        #delta
        cmp_cap[i][2] = random.randint(1,5)
        #Badwidth
        com_cap[i][0] = random.randint(1, 10)
        #power
        com_cap[i][1] = random.randint(20, 40)
        #gain
        dis= random.uniform(100,500)
        com_cap[i][2] = -128.1 -37.6*math.log10(dis)

    X, cmp_time = make_clients(args,n_samples=n_samples,cmp_cap=cmp_cap,com_cap=com_cap)
    n_clusters = centers


    improved_minmax = ImprovedMinMaxKMeans(n_clusters=n_clusters, beta=0, verbose=0)
    labels_minmax = improved_minmax.fit_predict(X,0)

    return labels_minmax, X, cmp_cap, com_cap, cmp_time


def clients_cluster(n_samples, n_clusters,args,clients_bds):
    # n_clusters=4
    # n_samples=64
    num_cluster = [0] * n_clusters
    min_num = min(num_cluster)
    client_list = [[] for _ in range(n_clusters)]
    while min_num<2:
        cluster_result, X, cmp_cap, com_cap, cmp_time = cluster(n_samples, n_clusters,args,clients_bds)
        if cluster_result is None:
            continue
        # print(com_cap)
        # print(cmp_cap)
        cluster_result = np.array(cluster_result).astype(dtype=int).tolist()
        count_list = cluster_result
        for j in range(n_clusters):
            num_cluster[j] = count_list.count(j)
        min_num = min(num_cluster)
        # max_num=max(num_cluster)
    #将计算时间最长的client作为簇头
    edge_list = [0]*n_clusters
    for j in range(n_clusters):
        # client_list=[]
        for i, x in enumerate(cluster_result):
            if x == j:
                client_list[j].append(i)
        max = client_list[j][0]
        for n in range(len(client_list[j])):
            if cmp_time[client_list[j][n]]>cmp_time[max]:
                max = client_list[j][n]
        edge_list[j]=max
        client_list[j].remove(max)
    # plt.figure()
    # colors = ['r','g','b','orange','yellow']
    # for n, y in enumerate(cluster_result):
    #     plt.plot(int(y), X[n], marker='.', color=colors[int(y)], ms=5)
    # # plt.title('Kmeans cluster centroids')
    # # plt.scatter(X[:, 0], X[:, 1], c=labels_minmax)
    # plt.title("Cluster assignments")
    # plt.show()
    return client_list, X, cmp_cap, com_cap, edge_list,cluster_result




