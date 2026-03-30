import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist, pdist

import matplotlib.cm as cm

class class_kmeans:
    def __init__(self, k=3, max_iters=10, interaction_flag=True):
        # ★ 하이퍼파라미터 설정 ★
        self.K = k # 클러스터의 개수 (사용자가 정해야 함, K-Means의 단점)
        self.max_iters = max_iters # 최대 반복 횟수 (무한루프 방지용)
        
        self.interaction_flag = interaction_flag  # 시각화 옵션: True면 그래프를 하나씩 띄우고 엔터를 쳐야 넘어감
        
        # 데이터와 결과물을 저장할 변수들 초기화 (아직은 None) 
        self.X, self.y = None, None # X: 입력 데이터(좌표), y: 정답 라벨(비지도라 안 쓰지만 생성함수에서 줌)
        self.centroids = None # 중심점 (Centroids) 좌표 저장소
        self.clusters = None # 각 점이 몇 번 클러스터인지 저장하는 배열 (0, 1, 2...)
        
        self.SSE, self.silhouette, self.dunn_index = None, None, None # 평가 지표 변수들
        
    def init_dataset(self, data_type='blobs'):
        if data_type == 'blobs':
            # make_blobs: 중심점(centers)을 기준으로 정규분포를 따르는 데이터 생성
            # cluster_std: 퍼짐 정도 (클수록 덩어리가 흐트러짐)
            self.X, _ = make_blobs(n_samples=300, centers=self.K, cluster_std=0.7, random_state=0)
        elif data_type == 'moons':
            # make_moons: 두 개의 초승달이 맞물린 형태
            self.X, self.y = make_moons(n_samples=300, noise=0.1, random_state=0)
        elif data_type == 'circles':
            # make_circles: 큰 원 안에 작은 원이 있는 형태
            self.X, self.y = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=0)
        else:
            # 기본값은 blobs
            self.X, _ = make_blobs(n_samples=300, centers=self.K, cluster_std=0.7, random_state=0)
       

    def initialize_centroids(self):
        """
        [Step 1] 초기 중심점 설정 (Random Initialization)
        - 데이터 중에서 무작위로 K개를 뽑아서 '초기 대장'으로 삼는다.
        """
        # np.random.choice: 0부터 데이터 개수(300) 사이에서 K개의 인덱스를 뽑음 (replace=False: 중복 불가)
        indices = np.random.choice(self.X.shape[0], self.K, replace=False)
        self.centroids = self.X[indices] # 뽑힌 인덱스의 좌표를 중심점으로 저장
        
        fig, ax = plt.subplots()
        colors = cm.tab10.colors  # 최대 10개의 색상을 지원하는 colormap
        for k in range(self.K):
            ax.scatter(self.X[:, 0], self.X[:, 1], s=30, c='gray', alpha=0.5)
        color = colors[k % len(colors)]    
        ax.scatter(self.centroids[:, 0], self.centroids[:, 1], s=400, c=[color], marker='X', edgecolor='k', linewidth=2, alpha=0.9)
        ax.set_title('Initial centroids')
        plt.show()
        
        if self.interaction_flag:
            input("Press Enter to start iterations...")


    def assign_cluster_index(self):
        """
        [Step 2] 할당 (Assignment) 단계
        - 모든 점에 대해, 가장 가까운 중심점(Centroid)을 찾아 소속을 정해준다.
        """
        clusters = []
        for point in self.X:
            # 1. 거리 계산: 현재 점(point)과 모든 중심점(self.centroids) 사이의 유클리드 거리 계산
            # np.linalg.norm: 벡터의 길이(거리)를 구하는 함수. (x1-x2)^2 + (y1-y2)^2 의 루트
            distances = np.linalg.norm(point - self.centroids, axis=1)
            # 2. 소속 결정: 거리가 가장 짧은(최소값) 인덱스를 찾음 (argmin)
            cluster_index = np.argmin(distances)
            clusters.append(cluster_index)
        self.clusters = np.array(clusters) # 결과를 numpy 배열로 저장
 
        
    def update_centroids(self):
        """
        [Step 3] 업데이트 (Update) 단계
        - 각 클러스터에 속한 점들의 '평균 위치(Mean)'로 깃발(중심점)을 옮긴다.
        """
        # 리스트 컴프리헨션(한 줄 코딩)으로 구현됨:
        # 1. range(self.K): 클러스터 0번, 1번, 2번... 순서대로
        # 2. self.X[self.clusters == i]: 현재 i번 클러스터에 속한 점들만 골라냄 (Boolean Indexing)
        # 3. .mean(axis=0): 그 점들의 X좌표 평균, Y좌표 평균을 구함 -> 새로운 중심점 좌표
        self.centroids = np.array([self.X[self.clusters == i].mean(axis=0) for i in range(self.K)])


    def has_converged(self, old_centroids):
        """
        [Step 4] 종료 조건 확인 (Convergence Check)
        - 중심점이 더 이상 움직이지 않으면(매우 조금 움직이면) 멈춘다.
        """
        tolerance = 1e-4

        # 이전 중심점(old)과 현재 중심점(self.centroids) 사이의 거리를 계산
        distances = np.linalg.norm(self.centroids - old_centroids, axis=1)
        # np.all: 모든 중심점의 이동 거리가 tolerance보다 작으면 True 반환
        return np.all(distances < tolerance)


    def kmeans_main(self):
        """
        ★ K-Means 알고리즘의 전체 실행 흐름 (Main Loop) ★
        """
        # 1. 초기화 (랜덤으로 중심점 뿌리기)
        self.initialize_centroids()
        
        # 2. 반복 (Iteration)
        for i in range(self.max_iters):
            # [단계 A] 할당: 각 점을 가장 가까운 대장에게 보냄
            self.assign_cluster_index()
            
            # (시각화: 할당된 상태 보여주기)
            self.plot_kmeans_plot(title=f'Iteration {i + 1} - After Assignment')
            
            if self.interaction_flag:
                input(f"Iteration {i + 1}: cluster index reassignment done...")            
            
            # 중심점 이동 전 위치 기억 (종료 조건 비교용)
            old_centroids = self.centroids

            # [단계 B] 업데이트: 대장을 무리의 한가운데로 이동시킴
            self.update_centroids()

            # (시각화: 대장이 이동한 상태 보여주기)
            self.plot_kmeans_plot(title=f'Iteration {i + 1} - After Centroid Update')
            if self.interaction_flag:
                input(f"Iteration {i + 1}: centroid update done...")            
            
            # [단계 C] 종료 조건 검사: 더 이상 안 움직이면 그만!
            if self.has_converged(old_centroids):
                print(f"\nK-means converged after {i+1} iterations")
                break


    def plot_kmeans_plot(self, title='kmeans result'):
        """
        현재 클러스터링 상태를 산점도(Scatter Plot)로 그려주는 함수
        """
        fig, ax = plt.subplots()
        colors = cm.tab10.colors  # 최대 10개의 색상을 지원하는 colormap
        # 1. 각 클러스터 별로 점 찍기
        for k in range(self.K):
            # 현재 k번 클러스터인 점들만 골라냄
            points = self.X[self.clusters == k]
            color = colors[k % len(colors)]
            # 점 그리기 (s=30: 점 크기, alpha=0.6: 투명도)
            ax.scatter(points[:, 0], points[:, 1], s=30, c=[color], label=f'Cluster {k+1}', alpha=0.6)
        
        # 중심점의 색상을 각 클러스터 색상에 맞추어 개별적으로 설정
        for idx, centroid in enumerate(self.centroids):
            # 중심점은 'X' 마크로 크게(s=400) 그림
            ax.scatter(centroid[0], centroid[1], s=400, c=[colors[idx % len(colors)]], 
                       marker='X', edgecolor='k', linewidth=2, alpha=0.9)
            
        ax.set_title(title)
        ax.legend()
        plt.show()


    def calculate_sse(self):
        """
        [평가 1] SSE (Sum of Squared Errors) - 오차 제곱 합
        - 의미: "우리 반 애들이 반장한테 얼마나 옹기종기 모여있나?"
        - 값이 작을수록 좋음 (응집도가 높음)
        """
        sse = 0.0
        for cluster_id in range(self.K):
            # 해당 클러스터의 점들만 추출
            cluster_points = self.X[self.clusters == cluster_id]

            # 혹시 중심점이 없으면 계산해서 가져옴 (예외 처리)
            if self.centroids is not None and len(self.centroids) > 0:
                centroid = self.centroids[cluster_id]
            else:
                self.update_centroids()
                centroid = self.centroids[cluster_id]

            # (점 위치 - 중심점 위치)의 제곱을 모두 더함
            # np.sum(... ** 2) -> 거리의 제곱 합
            sse += np.sum((cluster_points - centroid) ** 2)
        return sse
    
    

    def calculate_silhouette_score(self):
        """
        [평가 2] 실루엣 점수 (Silhouette Score)
        - 의미: "내 구역이랑은 가깝고, 옆 구역이랑은 먼가?"
        - 범위: -1 ~ 1 (1에 가까울수록 완벽하게 잘 나뉨)
        - sklearn 라이브러리 함수를 그대로 사용
        """
        return silhouette_score(self.X, self.clusters)


    def calculate_dunn_index(self):
        # 클러스터 간 최단 거리 (inter-cluster distance)
        """
        [평가 3] Dunn Index (던 지수) ★ 시험 출제 포인트
        - 공식: (군집 간 거리의 최소값) / (군집 내 거리의 최대값)
        - 의미: "끼리끼리는 똘똘 뭉치고(분모 작음), 남남끼리는 멀리 떨어져라(분자 큼)"
        - 값이 클수록 좋음
        """
        # [분자 계산] 군집 간 최단 거리 (Inter-cluster distance) 구하기
        min_inter_cluster_dist = np.inf # 무한대로 초기화 (최솟값 찾기 위함)
        for i in range(self.K):
            for j in range(i + 1, self.K): # 자기 자신 제외하고 다른 클러스터와 비교
                # 중심점끼리의 거리 계산
                inter_cluster_dist = np.linalg.norm(self.centroids[i] - self.centroids[j])
                if inter_cluster_dist < min_inter_cluster_dist:
                    min_inter_cluster_dist = inter_cluster_dist # 더 짧은 거리가 나오면 갱신

        # [분모 계산] # 클러스터 내 최대 거리 (intra-cluster distance)
        max_intra_cluster_dist = 0 # [분모 계산] 군집 내 최대 거리 (Intra-cluster distance) 구하기
        for cluster_id in range(self.K): # 자기 자신 제외하고 다른 클러스터와 비교
            
            # 중심점끼리의 거리 계산
            cluster_points = self.X[self.clusters == cluster_id]
            # 여기서는 (군집 내 모든 점) <-> (자기네 중심점) 사이 거리를 잰 뒤 가장 큰 놈을 찾음
            intra_cluster_dist = np.max(cdist(cluster_points, [self.centroids[cluster_id]], metric='euclidean'))
            if intra_cluster_dist > max_intra_cluster_dist: # 더 큰 거리가 나오면 갱신
                max_intra_cluster_dist = intra_cluster_dist 

        # Dunn Index 계산
        dunn_index = min_inter_cluster_dist / max_intra_cluster_dist
        return dunn_index 




    def evaluate(self):

        self.sse = self.calculate_sse()
        self.silhouette = self.calculate_silhouette_score()
        self.dunn_index = self.calculate_dunn_index()
        
        print("\nSum of Squared Errors (SSE):", self.sse)
        print("Silhouette Score:", self.silhouette)
        print("Dunn Index:", self.dunn_index)



def blobs_main():
    ck = class_kmeans(k=3, interaction_flag=False)
    ck.init_dataset(data_type='blobs')
    ck.kmeans_main()
    ck.evaluate()
    
def moons_main():
    ck = class_kmeans(k=2, interaction_flag=False)
    ck.init_dataset(data_type='moons')
    ck.kmeans_main()
    ck.evaluate()    
    
def circles_main():
    ck = class_kmeans(k=2, interaction_flag=False)
    ck.init_dataset(data_type='circles')
    ck.kmeans_main()
    ck.evaluate()        

if __name__ == '__main__':

    blobs_main()
    #moons_main()
    #circles_main()
    
    
