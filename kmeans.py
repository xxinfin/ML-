import numpy as np


class Kmeans:
    def __init__(self,data,num_cluster):
        self.data=data
        self.num_cluster=num_cluster
    def train(self,max_itertions):
        #1.初始化质心。先随机找到中心点
        centroids=Kmeans.centroids_init(self.data,self.num_cluster)
       #计算每个点到所有质心点的距离
        num_examples=self.data.shape[0]
        closest_centrioids_ids=np.empty((num_examples,1))
        #开始训练
        for _ in range(max_itertions):
            #当前每个样本点到K 个中心点的距离,找到最近的中心点
            closest_centrioids_ids=Kmeans.centroids_find_closest(self.data,centroids)
            #质心参数的更新
            centroids=Kmeans.centroids_compute(self.data,closest_centrioids_ids,self.num_cluster)
        return centroids,closest_centrioids_ids
    @staticmethod
    def centroids_init(data,num_cluster):
        num_examples=data.shape[0]
        random_ids=np.random.permutation(num_examples)
        centroids=data[random_ids[:num_cluster],:]
        return centroids
    @staticmethod
    def centroids_find_closest(data,centroids):
        #距离计算方法，欧氏距离
        num_example=data.shape[0]
        num_centroids=centroids.shape[0]
        closest_centrioids=np.zeros((num_example,1))
        for example_index in range(num_example):
            distance=np.zeros(num_centroids)
            for  centroid_index in range(num_centroids):
                distance_diff=data[example_index,:]-centroids[centroid_index,:]
                distance[centroid_index]=np.sum(distance_diff**2)
            closest_centrioids[example_index]=np.argmin(distance)
        return closest_centrioids
    @staticmethod
    def centroids_compute(data, closest_centrioids_ids,num_clustes):
        #均值计算
        num_features=data.shape[1]
        centorids=np.zeros((num_clustes,num_features))
        for centorids_id in range(num_clustes):
            closest_ids=closest_centrioids_ids==centorids_id
            centorids[centorids_id]=np.mean(data[closest_ids.flatten(),:],axis=0)
        return centorids


