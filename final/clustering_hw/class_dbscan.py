import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import matplotlib.cm as cm
import hdbscan

class class_dbscan:
    def __init__(self):
        
        self.X, self.y = None, None
        self.centroids = None
        self.clusters = None
        
        self.SSE, self.silhouette, self.dunn_index = None, None, None
        
    def init_dataset(self, data_type='blobs'):
        if data_type == 'blobs':
            self.X, self.y = make_blobs(n_samples=300, centers=3, cluster_std=0.7, random_state=0)
        elif data_type == 'moons':
            self.X, self.y = make_moons(n_samples=300, noise=0.1, random_state=0)
        elif data_type == 'circles':
            self.X, self.y = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=0)
        else:
            self.X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.7, random_state=0)
            

            
    def dbscan_main(self, eps=0.2, min_samples=5):
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.clusters = dbscan.fit_predict(self.X)     
        self.K = len(set(self.clusters)) - (1 if -1 in self.clusters else 0)

        plot_title = f"eps={eps}, min_samples={min_samples}"
        self.plot_dbscan_plot(title=plot_title)


    def hdbscan_main(self, min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.0):
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,                 # None이면 min_cluster_size 기준으로 자동 설정
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric='euclidean'
        )
    
        # 클러스터링 수행
        self.clusters = clusterer.fit_predict(self.X)
    
        # 군집 개수 계산 (라벨 -1은 noise)
        self.K = len(set(self.clusters)) - (1 if -1 in self.clusters else 0)
    
        plot_title = (
            f"HDBSCAN: min_cluster_size={min_cluster_size}, "
            f"min_samples={min_samples}, "
            f"cluster_selection_epsilon={cluster_selection_epsilon}"
        )
        self.plot_dbscan_plot(title=plot_title)



    def plot_dbscan_plot(self, title='dbscan result'):
        fig, ax = plt.subplots()
        # 색상을 클러스터 수에 맞게 자동으로 생성
        colors = cm.tab10.colors  # tab10은 최대 10개의 색상을 제공
        for k in range(self.K):
            color = colors[k % len(colors)]  # 클러스터 수가 10을 넘을 경우 색상을 반복 사용
            points = self.X[self.clusters == k]
            ax.scatter(points[:, 0], points[:, 1], s=30, c=[color], label=f'Cluster {k+1}', alpha=0.6)
        
        # 노이즈 포인트를 회색으로 표시
        noise_points = self.X[self.clusters == -1]
        ax.scatter(noise_points[:, 0], noise_points[:, 1], s=30, c='black', label='Noise', alpha=0.6)
        
        ax.set_title(title)
        ax.legend()
        plt.show()


    def update_centroids(self):
        # 각 군집의 중심점을 계산합니다.
        self.centroids = np.array([self.X[self.clusters == i].mean(axis=0) for i in range(self.K)])

    def calculate_sse(self):
        sse = 0.0
        for cluster_id in range(self.K):
            cluster_points = self.X[self.clusters == cluster_id]
            if self.centroids is not None and len(self.centroids) > 0:
                centroid = self.centroids[cluster_id]
            else:
                self.update_centroids()
                centroid = self.centroids[cluster_id]
                
            sse += np.sum((cluster_points - centroid) ** 2)
        return sse
    
    
    
    def calculate_silhouette_score(self):
        labels = self.clusters
        X = self.X
        
        # 노이즈(-1) 제거
        mask = (labels != -1)
        filtered_X = X[mask]
        filtered_labels = labels[mask]
        
        unique_labels = set(filtered_labels)
        n_clusters = len(unique_labels)
    
        if n_clusters > 1:
            score = silhouette_score(filtered_X, filtered_labels)
        else:
            print("클러스터가 하나뿐입니다. Silhouette score를 계산할 수 없습니다.")
            score = 0
        
        return score

    
    def calculate_dunn_index(self):
        """
        DBSCAN 결과에 대해 centroid 기반 Dunn Index를 계산.
        - 노이즈 라벨(-1)은 자동으로 제외함.
        - inter-cluster: 클러스터 중심 간 최소 거리
        - intra-cluster: 각 클러스터 내 점-센터 최대 거리
        """
        X = self.X
        labels = self.clusters
        centroids = self.centroids
    
        #1. 노이즈(-1) 라벨 제외
        valid_mask = labels != -1
        X_valid = X[valid_mask]
        labels_valid = labels[valid_mask]
    
        unique_labels = np.unique(labels_valid)
        K_valid = len(unique_labels)
    
        if K_valid < 2:
            print("클러스터가 2개 미만입니다. Dunn Index 계산 불가.")
            return 0.0
    
        #2.실제 유효한 클러스터의 centroid만 추출
        #    (DBSCAN이라면 self.centroids는 미리 계산되어 있어야 함)
        valid_centroids = np.array([centroids[c] for c in unique_labels])
    
        #3.클러스터 간 최소 중심 거리 (inter-cluster distance)
        min_inter_cluster_dist = np.inf
        for i in range(K_valid):
            for j in range(i + 1, K_valid):
                dist = np.linalg.norm(valid_centroids[i] - valid_centroids[j])
                if dist < min_inter_cluster_dist:
                    min_inter_cluster_dist = dist
    
        #4. 클러스터 내 최대 점-센터 거리 (intra-cluster distance)
        max_intra_cluster_dist = 0
        for cluster_id in unique_labels:
            cluster_points = X_valid[labels_valid == cluster_id]
            if cluster_points.size > 0:
                # 점-센터 최대 거리 계산
                dists = cdist(cluster_points, [centroids[cluster_id]], metric='euclidean')
                intra_dist = np.max(dists)
                if intra_dist > max_intra_cluster_dist:
                    max_intra_cluster_dist = intra_dist
    
        #5. Dunn Index 계산
        if max_intra_cluster_dist == 0:
            return float('inf')  # 완벽 응집 시
        else:
            return float(min_inter_cluster_dist / max_intra_cluster_dist)


    def evaluate(self):
        self.sse = self.calculate_sse()
        self.silhouette = self.calculate_silhouette_score()
        self.dunn_index = self.calculate_dunn_index()
        
        print(f"\nSum of Squared Errors (SSE): {self.sse}")
        print(f"Silhouette Score: {self.silhouette}")
        print(f"Dunn Index: {self.dunn_index}")


def dbscan_blobs_main():
    cb = class_dbscan()
    cb.init_dataset(data_type='blobs')
    cb.dbscan_main(eps=0.6, min_samples=10)
    cb.evaluate()


def dbscan_blobs_variable_main():
    cb = class_dbscan()
    cb.init_dataset(data_type='blobs')
    eps_list = np.arange(0.2, 1, 0.2)
    min_samples_list = np.arange(4, 15, 2)
    for eps in eps_list:
        for min_samples in min_samples_list:
            cb.dbscan_main(eps=eps, min_samples=min_samples)
            cb.evaluate()

def hdbscan_blobs_main():
    cb = class_dbscan()
    cb.init_dataset(data_type='blobs')
    cb.hdbscan_main()
    cb.evaluate()

def dbscan_moons_main():
    cb = class_dbscan()
    cb.init_dataset(data_type='moons')
    cb.dbscan_main(eps=0.2, min_samples=5)
    cb.evaluate()


def dbscan_circles_main():
    cb = class_dbscan()
    cb.init_dataset(data_type='circles')
    cb.dbscan_main(eps=0.2, min_samples=5)
    cb.evaluate()
    
if __name__ == '__main__':

    dbscan_blobs_main()
    #dbscan_moons_main()
    #dbscan_circles_main()

    #dbscan_blobs_variable_main()
    #hdbscan_blobs_main()