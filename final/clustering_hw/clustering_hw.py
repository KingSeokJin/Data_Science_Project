import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

# 머신러닝/데이터 처리 관련 라이브러리
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
import hdbscan

def load_data():
    """
    데이터셋 로드 및 전처리 함수 (class_image_clustering의 로직 통합)
    """
    # 현재 파일 위치 기준 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    embedding_file = os.path.join(base_dir, "image_embedding.npz")
    tar_dir = os.path.join(base_dir, 'image_samples')

    if not os.path.exists(embedding_file):
        raise FileNotFoundError(f"임베딩 파일을 찾을 수 없습니다: {embedding_file}")

    data = np.load(embedding_file, allow_pickle=True)
    filenames = data['filenames']
    image_embedding = data['image_embedding']
    
    # 인덱스 매핑 생성
    idx2filename = {idx: fn for idx, fn in enumerate(filenames)}
    
    print(f"임베딩 로드 완료: {len(image_embedding)}개")
    
    # 정규화 수행 (K-Means, DBSCAN 등 거리 기반 알고리즘을 위해 필수)
    X_normalized = normalize(image_embedding, axis=1)
    
    return X_normalized, idx2filename, tar_dir

def reduce_dimensions(X, n_components=2, method='pca', random_state=42):
    """차원 축소 (PCA 또는 t-SNE)"""
    print(f"차원 축소 수행 중 (Method: {method})...")
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_state)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=random_state, 
                       perplexity=30, n_iter=300)
    else:
        raise ValueError("지원하지 않는 차원 축소 방법입니다: 'pca', 'tsne'")
    
    X_reduced = reducer.fit_transform(X)
    print("차원 축소 완료.")
    return X_reduced

def plot_clusters_2d(X_2d, labels, algorithm_name, params_str, metrics_dict, dim_reduction_method='PCA'):
    """2D 클러스터링 결과 시각화"""
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    colors = cm.tab20(np.linspace(0, 1, max(10, n_clusters))) 
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 노이즈 포인트 (lightgray)
    if -1 in unique_labels:
        noise_mask = (labels == -1)
        ax.scatter(X_2d[noise_mask, 0], X_2d[noise_mask, 1], 
                   s=10, c='lightgray', label='noise', alpha=0.7)

    # 클러스터 포인트
    cluster_labels = sorted([l for l in unique_labels if l != -1])
    for k in cluster_labels:
        cluster_mask = (labels == k)
        color = colors[k % len(colors)]
        ax.scatter(X_2d[cluster_mask, 0], X_2d[cluster_mask, 1], 
                   s=15, color=color, label=f'cluster {k}', alpha=0.9)
    
    # 제목 및 범례
    title = f"{algorithm_name} ({params_str}) {dim_reduction_method} 2D"
    ax.set_title(title, fontsize=16)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # 메트릭 정보 표시
    metrics_text = (
        f"**Results**\n"
        f"Clusters found: {n_clusters}\n\n"
        f"**Metrics**\n"
        f"SSE: {metrics_dict.get('sse', 'N/A'):.2f}\n"
        f"Silhouette: {metrics_dict.get('silhouette', 'N/A'):.4f}\n"
        f"Dunn Index: {metrics_dict.get('dunn', 'N/A'):.4f}"
    )
    
    ax.text(1.05, 0.7, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.3))
    
    fig.tight_layout(rect=[0, 0, 0.8, 1])
    plt.show()

def show_representative_images(X_data, labels, idx2filename, tar_dir, n_images=6):
    """각 클러스터의 대표 이미지 시각화"""
    unique_labels = sorted([l for l in set(labels) if l != -1])
    
    if not unique_labels:
        print("유효한 클러스터가 없어 대표 이미지를 표시할 수 없습니다.")
        return

    for cluster_id in unique_labels:
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0: continue
            
        cluster_embeddings = X_data[cluster_indices]
        centroid = np.mean(cluster_embeddings, axis=0)
        
        distances = cdist(cluster_embeddings, [centroid], metric='euclidean')
        closest_indices_in_cluster = np.argsort(distances.ravel())[:n_images]
        original_indices = cluster_indices[closest_indices_in_cluster]
        
        image_files = [os.path.join(tar_dir, idx2filename[idx]) for idx in original_indices]
        
        fig, axs = plt.subplots(1, n_images, figsize=(15, 3))
        if n_images == 1: axs = [axs]
            
        fig.suptitle(f"Cluster {cluster_id} - Representative Images", fontsize=14)
        
        for i, img_path in enumerate(image_files):
            try:
                img = Image.open(img_path)
                axs[i].imshow(img)
                axs[i].axis('off')
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                axs[i].axis('off')
        
        for j in range(len(image_files), n_images):
            axs[j].axis('off')
        plt.show()

def calculate_sse_metric(X, labels, centroids=None):
    """SSE 계산"""
    sse = 0.0
    unique_labels = [l for l in np.unique(labels) if l != -1] # 노이즈 제외
    
    if centroids is None:
        # 중심점이 없으면 계산 (DBSCAN 등)
        centroids = {}
        for label in unique_labels:
            centroids[label] = np.mean(X[labels == label], axis=0)
    elif isinstance(centroids, np.ndarray):
        # K-Means 처럼 배열로 들어오면 딕셔너리로 변환 (편의상)
        centroids_dict = {i: centroids[i] for i in range(len(centroids))}
        centroids = centroids_dict

    for label in unique_labels:
        cluster_points = X[labels == label]
        centroid = centroids[label]
        sse += np.sum((cluster_points - centroid) ** 2)
    return sse

def calculate_dunn_index_metric(X, labels, centroids=None):
    """
    Dunn Index 계산 (class_dbscan.py의 로직 기반으로 일반화)
    """
    # 노이즈(-1) 제외
    valid_mask = labels != -1
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]
    unique_labels = np.unique(labels_valid)
    K_valid = len(unique_labels)

    if K_valid < 2:
        return 0.0

    # 중심점 준비
    if centroids is None:
        centroids = np.array([np.mean(X_valid[labels_valid == l], axis=0) for l in unique_labels])
    elif isinstance(centroids, np.ndarray) and len(centroids) != K_valid:
         # K-Means의 경우 초기 K개와 실제 할당된 클러스터 개수가 다를 수 있으므로 다시 계산
         centroids = np.array([np.mean(X_valid[labels_valid == l], axis=0) for l in unique_labels])

    # 1. 클러스터 간 최소 중심 거리 (inter-cluster distance)
    min_inter_cluster_dist = np.inf
    for i in range(K_valid):
        for j in range(i + 1, K_valid):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            if dist < min_inter_cluster_dist:
                min_inter_cluster_dist = dist

    # 2. 클러스터 내 최대 점-센터 거리 (intra-cluster distance)
    max_intra_cluster_dist = 0
    for i, label in enumerate(unique_labels):
        cluster_points = X_valid[labels_valid == label]
        if cluster_points.size > 0:
            dists = cdist(cluster_points, [centroids[i]], metric='euclidean')
            intra_dist = np.max(dists)
            if intra_dist > max_intra_cluster_dist:
                max_intra_cluster_dist = intra_dist

    if max_intra_cluster_dist == 0:
        return float('inf')
    else:
        return min_inter_cluster_dist / max_intra_cluster_dist


class CustomKMeans:
    def __init__(self, k=3, max_iters=10):
        self.K = k
        self.max_iters = max_iters
        self.X = None
        self.centroids = None
        self.clusters = None
        self.sse = None
        self.silhouette = None
        self.dunn_index = None

    def initialize_centroids(self):
        indices = np.random.choice(self.X.shape[0], self.K, replace=False)
        self.centroids = self.X[indices]

    def assign_cluster_index(self):
        clusters = []
        for point in self.X:
            distances = np.linalg.norm(point - self.centroids, axis=1)
            cluster_index = np.argmin(distances)
            clusters.append(cluster_index)
        self.clusters = np.array(clusters)

    def update_centroids(self):
        new_centroids = []
        for i in range(self.K):
            points_in_cluster = self.X[self.clusters == i]
            if len(points_in_cluster) > 0:
                new_centroids.append(points_in_cluster.mean(axis=0))
            else:
                # 클러스터에 점이 하나도 없으면 기존 중심점 유지
                new_centroids.append(self.centroids[i])
        self.centroids = np.array(new_centroids)

    def has_converged(self, old_centroids):
        tolerance = 1e-4
        distances = np.linalg.norm(self.centroids - old_centroids, axis=1)
        return np.all(distances < tolerance)

    def fit(self, X):
        self.X = X
        self.initialize_centroids()

        for i in range(self.max_iters):
            self.assign_cluster_index()
            old_centroids = self.centroids
            self.update_centroids()
            
            if self.has_converged(old_centroids):
                print(f"K-means converged after {i+1} iterations")
                break
        
        # 최종 평가 지표 계산
        self.evaluate()

    def evaluate(self):
        self.sse = calculate_sse_metric(self.X, self.clusters, self.centroids)
        # 실루엣 스코어는 최소 2개 이상의 클러스터가 있어야 함
        if len(np.unique(self.clusters)) > 1:
            self.silhouette = silhouette_score(self.X, self.clusters)
        else:
            self.silhouette = 0
        self.dunn_index = calculate_dunn_index_metric(self.X, self.clusters, self.centroids)

class CustomClusteringWrapper:
    """DBSCAN과 HDBSCAN을 수행하고 평가하기 위한 래퍼"""
    def __init__(self):
        self.X = None
        self.clusters = None
        self.sse = None
        self.silhouette = None
        self.dunn_index = None

    def run_dbscan(self, X, eps=0.2, min_samples=5):
        self.X = X
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.clusters = dbscan.fit_predict(self.X)
        self.evaluate()

    def run_hdbscan(self, X, min_cluster_size=5, min_samples=None):
        self.X = X
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean'
        )
        self.clusters = clusterer.fit_predict(self.X)
        self.evaluate()

    def evaluate(self):
        self.sse = calculate_sse_metric(self.X, self.clusters) # Centroid 자동 계산됨
        
        # 노이즈만 있거나 클러스터가 1개 미만인 경우 처리
        unique_labels = set(self.clusters)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        if n_clusters > 1:
            # 실루엣 점수 계산 시 노이즈 제외
            mask = self.clusters != -1
            if np.sum(mask) > 1: # 유효한 데이터가 2개 이상이어야 함
                self.silhouette = silhouette_score(self.X[mask], self.clusters[mask])
            else:
                self.silhouette = 0
        else:
            self.silhouette = 0
            
        self.dunn_index = calculate_dunn_index_metric(self.X, self.clusters)

if __name__ == '__main__':
    
    # --- 0. 데이터 로드 ---
    print("Step 0: 이미지 임베딩 데이터 로드 및 전처리...")
    X_data, idx2filename, tar_dir = load_data()
    
    # --- 시각화를 위한 2D 차원 축소 (PCA) ---
    X_reduced_2d = reduce_dimensions(X_data, n_components=2, method='pca')
    
    results_summary = {}

    # --- 1. K-Means ---
    print("\n--- Step 1: K-Means 클러스터링 수행 ---")
    K_OPTIMAL = 10
    params_str_kmeans = f"K={K_OPTIMAL}"
    print(f"선택된 파라미터: {params_str_kmeans}")

    # [요구사항 반영] import 없이 자체 클래스 사용, 시각화 과정 생략됨
    kmeans = CustomKMeans(k=K_OPTIMAL)
    kmeans.fit(X_data) # 내부적으로 evaluate 수행

    print("K-Means 평가:")
    print(f"SSE: {kmeans.sse}")
    print(f"Silhouette: {kmeans.silhouette}")
    print(f"Dunn Index: {kmeans.dunn_index}")

    kmeans_metrics = {'sse': kmeans.sse, 'silhouette': kmeans.silhouette, 'dunn': kmeans.dunn_index}
    results_summary['K-Means'] = kmeans_metrics
    
    plot_clusters_2d(X_reduced_2d, kmeans.clusters, "K-Means", params_str_kmeans, kmeans_metrics)
    show_representative_images(X_data, kmeans.clusters, idx2filename, tar_dir)


    # --- 2. DBSCAN ---
    print("\n--- Step 2: DBSCAN 클러스터링 수행 ---")
    MIN_SAMPLES_OPTIMAL = 5 
    EPS_OPTIMAL = 0.6 
    params_str_dbscan = f"eps={EPS_OPTIMAL}, min_samples={MIN_SAMPLES_OPTIMAL}"
    print(f"선택된 파라미터: {params_str_dbscan}")

    dbscan_wrapper = CustomClusteringWrapper()
    dbscan_wrapper.run_dbscan(X_data, eps=EPS_OPTIMAL, min_samples=MIN_SAMPLES_OPTIMAL)

    print("DBSCAN 평가:")
    print(f"SSE: {dbscan_wrapper.sse}")
    print(f"Silhouette: {dbscan_wrapper.silhouette}")
    print(f"Dunn Index: {dbscan_wrapper.dunn_index}")

    dbscan_metrics = {'sse': dbscan_wrapper.sse, 'silhouette': dbscan_wrapper.silhouette, 'dunn': dbscan_wrapper.dunn_index}
    results_summary['DBSCAN'] = dbscan_metrics

    plot_clusters_2d(X_reduced_2d, dbscan_wrapper.clusters, "DBSCAN", params_str_dbscan, dbscan_metrics)
    show_representative_images(X_data, dbscan_wrapper.clusters, idx2filename, tar_dir)


    # --- 3. HDBSCAN ---
    print("\n--- Step 3: HDBSCAN 클러스터링 수행 ---")
    MIN_CLUSTER_SIZE_OPTIMAL = 10
    MIN_SAMPLES_HDBSCAN = None 
    params_str_hdbscan = f"min_cluster_size={MIN_CLUSTER_SIZE_OPTIMAL}"
    print(f"선택된 파라미터: {params_str_hdbscan}")

    hdbscan_wrapper = CustomClusteringWrapper()
    hdbscan_wrapper.run_hdbscan(X_data, min_cluster_size=MIN_CLUSTER_SIZE_OPTIMAL, min_samples=MIN_SAMPLES_HDBSCAN)

    print("HDBSCAN 평가:")
    print(f"SSE: {hdbscan_wrapper.sse}")
    print(f"Silhouette: {hdbscan_wrapper.silhouette}")
    print(f"Dunn Index: {hdbscan_wrapper.dunn_index}")

    hdbscan_metrics = {'sse': hdbscan_wrapper.sse, 'silhouette': hdbscan_wrapper.silhouette, 'dunn': hdbscan_wrapper.dunn_index}
    results_summary['HDBSCAN'] = hdbscan_metrics

    plot_clusters_2d(X_reduced_2d, hdbscan_wrapper.clusters, "HDBSCAN", params_str_hdbscan, hdbscan_metrics)
    show_representative_images(X_data, hdbscan_wrapper.clusters, idx2filename, tar_dir)


    # --- 4. 최종 결과 비교 ---
    print("\n--- Step 4: 최종 클러스터링 성능 비교 ---")
    print(f"{'Algorithm':<12} | {'SSE (낮을수록 좋음)':<20} | {'Silhouette (1에 가까울수록)':<25} | {'Dunn Index (높을수록 좋음)':<25}")
    print("-" * 85)
    
    for algo, metrics in results_summary.items():
        print(f"{algo:<12} | {metrics['sse']:<20.2f} | {metrics['silhouette']:<25.4f} | {metrics['dunn']:<25.4f}")