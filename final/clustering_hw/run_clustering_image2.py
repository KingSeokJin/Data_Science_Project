import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import os

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

# 제공된 클래스 파일 임포트
from class_image_clustering import class_image_clustering
from class_kmeans import class_kmeans
from class_dbscan import class_dbscan

def reduce_dimensions(X, n_components=2, method='pca', random_state=42):
    """
    데이터를 저차원(2D 또는 3D)으로 축소합니다.
    method: 'pca', 'tsne'
    """
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
    """
    요청사항에 맞춘 2D 클러스터링 시각화 함수
    """
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    colors = cm.tab20(np.linspace(0, 1, max(10, n_clusters))) 
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 1. 노이즈 포인트 (lightgray)
    if -1 in unique_labels:
        noise_mask = (labels == -1)
        ax.scatter(X_2d[noise_mask, 0], X_2d[noise_mask, 1], 
                   s=10, c='lightgray', label='noise', alpha=0.7)

    # 2. 클러스터 포인트
    cluster_labels = sorted([l for l in unique_labels if l != -1])
    for k in cluster_labels:
        cluster_mask = (labels == k)
        color = colors[k % len(colors)]
        ax.scatter(X_2d[cluster_mask, 0], X_2d[cluster_mask, 1], 
                   s=15, color=color, label=f'cluster {k}', alpha=0.9)
    
    # 3. 제목 및 범례
    title = f"{algorithm_name} ({params_str}) {dim_reduction_method} 2D"
    ax.set_title(title, fontsize=16)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # 4. 메트릭 정보 표시
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
    
    fig.tight_layout(rect=[0, 0, 0.8, 1]) # 텍스트/범례 공간 확보
    plt.show()


def show_representative_images(X_data, labels, idx2filename, tar_dir, n_images=6):
    """
    각 클러스터의 중심(centroid)에 가장 가까운 대표 이미지들을 보여줍니다.
    """
    unique_labels = sorted([l for l in set(labels) if l != -1]) # 노이즈(-1) 제외
    
    if not unique_labels:
        print("유효한 클러스터가 없어 대표 이미지를 표시할 수 없습니다.")
        return

    for cluster_id in unique_labels:
        cluster_indices = np.where(labels == cluster_id)[0]
        
        if len(cluster_indices) == 0:
            continue
            
        cluster_embeddings = X_data[cluster_indices]
        centroid = np.mean(cluster_embeddings, axis=0)
        
        distances = cdist(cluster_embeddings, [centroid], metric='euclidean')
        closest_indices_in_cluster = np.argsort(distances.ravel())[:n_images]
        
        original_indices = cluster_indices[closest_indices_in_cluster]
        
        image_files = [os.path.join(tar_dir, idx2filename[idx]) for idx in original_indices]
        
        fig, axs = plt.subplots(1, n_images, figsize=(15, 3))
        if n_images == 1:
            axs = [axs]
            
        fig.suptitle(f"Cluster {cluster_id} - Representative Images ({len(cluster_indices)} items)", fontsize=14)
        
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


# --- 2. 메인 실행 로직 ---

if __name__ == '__main__':
    
    # --- 0. 데이터 로드 및 전처리 ---
    print("Step 0: 이미지 임베딩 데이터 로드 및 전처리...")
    
    ci = class_image_clustering()
    ci.load_embeddings_npz()
    
    image_embeddings = ci.image_embedding
    idx2filename = ci.idx2filename
    tar_dir = ci.tar_dir
    
    X_data = normalize(image_embeddings, axis=1)
    
    print(f"데이터 로드 완료: {X_data.shape[0]}개 샘플, {X_data.shape[1]}개 특성")

    # --- 시각화를 위한 2D 데이터 미리 생성 (PCA) ---
    X_reduced_2d = reduce_dimensions(X_data, n_components=2, method='pca')
    
    # 최종 비교를 위한 딕셔너리
    results_summary = {}

    
    # --- 1. K-Means 클러스터링 ---
    print("\n--- Step 1: K-Means 클러스터링 수행 ---")
    
    # (1) 하이퍼파라미터 설정
    K_OPTIMAL = 10
    params_str_kmeans = f"K={K_OPTIMAL}"
    print(f"선택된 파라미터: {params_str_kmeans}")
    
    # (2) K-Means 실행
    kmeans = class_kmeans(k=K_OPTIMAL, interaction_flag=False)
    kmeans.X = X_data # 로드한 데이터(X_data)를 클래스의 X 속성에 직접 할당
    
    
    # K-Means의 중간 플롯을 끄기 위한 임시 조치
    print("K-Means 실행 (중간 과정 플롯 비활성화)...")
    original_plt_show = plt.show  # 원래의 plt.show 함수를 백업
    plt.show = lambda: None       # plt.show를 아무것도 안 하는 빈 함수로 교체
    
    kmeans.kmeans_main() # 이 함수 안의 모든 plt.show()가 무시됨
    
    plt.show = original_plt_show  # 원래의 plt.show 함수로 복원
    # 임시 조치 끝

    # (3) 평가
    print("K-Means 평가:")
    kmeans.evaluate()
    kmeans_labels = kmeans.clusters
    kmeans_metrics = {
        'sse': kmeans.sse,
        'silhouette': kmeans.silhouette,
        'dunn': kmeans.dunn_index
    }
    results_summary['K-Means'] = kmeans_metrics
    
    # (4) 시각화 및 대표 이미지
    plot_clusters_2d(X_reduced_2d, kmeans_labels, "K-Means", params_str_kmeans, kmeans_metrics)
    show_representative_images(X_data, kmeans_labels, idx2filename, tar_dir)


    # --- 2. DBSCAN 클러스터링 ---
    print("\n--- Step 2: DBSCAN 클러스터링 수행 ---")
    
    # (1) 하이퍼파라미터 설정 (예시 이미지 및 k-distance plot 기반)
    MIN_SAMPLES_OPTIMAL = 5 
    EPS_OPTIMAL = 0.6 
    params_str_dbscan = f"eps={EPS_OPTIMAL}, min_samples={MIN_SAMPLES_OPTIMAL}"
    print(f"선택된 파라미터: {params_str_dbscan}")

    # (2) DBSCAN 실행
    dbscan = class_dbscan()
    dbscan.X = X_data # 로드한 데이터(X_data)를 클래스의 X 속성에 직접 할당
    dbscan.dbscan_main(eps=EPS_OPTIMAL, min_samples=MIN_SAMPLES_OPTIMAL)

    # (3) 평가
    print("DBSCAN 평가:")
    dbscan.evaluate()
    dbscan_labels = dbscan.clusters
    dbscan_metrics = {
        'sse': dbscan.sse,
        'silhouette': dbscan.silhouette,
        'dunn': dbscan.dunn_index
    }
    results_summary['DBSCAN'] = dbscan_metrics
    
    # (4) 시각화 및 대표 이미지
    plot_clusters_2d(X_reduced_2d, dbscan_labels, "DBSCAN", params_str_dbscan, dbscan_metrics)
    show_representative_images(X_data, dbscan_labels, idx2filename, tar_dir)


    # --- 3. HDBSCAN 클러스터링 ---
    print("\n--- Step 3: HDBSCAN 클러스터링 수행 ---")
    
    # (1) 하이퍼파라미터 설정
    MIN_CLUSTER_SIZE_OPTIMAL = 10
    MIN_SAMPLES_HDBSCAN = None # None으로 두면 min_cluster_size를 따라감
    
    params_str_hdbscan = f"min_cluster_size={MIN_CLUSTER_SIZE_OPTIMAL}"
    print(f"선택된 파라미터: {params_str_hdbscan}")

    # (2) HDBSCAN 실행
    hdbscan_clusterer = class_dbscan() # class_dbscan 클래스 내의 hdbscan_main 사용
    hdbscan_clusterer.X = X_data # 데이터 직접 할당
    hdbscan_clusterer.hdbscan_main(
        min_cluster_size=MIN_CLUSTER_SIZE_OPTIMAL,
        min_samples=MIN_SAMPLES_HDBSCAN
    )

    # (3) 평가
    print("HDBSCAN 평가:")
    hdbscan_clusterer.evaluate()
    hdbscan_labels = hdbscan_clusterer.clusters
    hdbscan_metrics = {
        'sse': hdbscan_clusterer.sse,
        'silhouette': hdbscan_clusterer.silhouette,
        'dunn': hdbscan_clusterer.dunn_index
    }
    results_summary['HDBSCAN'] = hdbscan_metrics
    
    # (4) 시각화 및 대표 이미지
    plot_clusters_2d(X_reduced_2d, hdbscan_labels, "HDBSCAN", params_str_hdbscan, hdbscan_metrics)
    show_representative_images(X_data, hdbscan_labels, idx2filename, tar_dir)


    # --- 4. 최종 결과 비교 ---
    print("\n--- Step 4: 최종 클러스터링 성능 비교 ---")
    
    print(f"{'Algorithm':<12} | {'SSE (낮을수록 좋음)':<20} | {'Silhouette (1에 가까울수록)':<25} | {'Dunn Index (높을수록 좋음)':<25}")
    print("-" * 85)
    
    for algo, metrics in results_summary.items():
        print(f"{algo:<12} | {metrics['sse']:<20.2f} | {metrics['silhouette']:<25.4f} | {metrics['dunn']:<25.4f}")