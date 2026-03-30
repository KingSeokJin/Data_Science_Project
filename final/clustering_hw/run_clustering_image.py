import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

from class_image_clustering import class_image_clustering
from sklearn.cluster import KMeans, DBSCAN
import hdbscan
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist


# ---------------------------
# 🔥 Silhouette 기반 K 자동 선택
# ---------------------------
def find_best_k(embeddings, k_min=3, k_max=15):
    best_k = None
    best_score = -1
    scores = {}

    print("\n=== Silhouette 기반 최적 K 탐색 ===")
    for k in range(k_min, k_max + 1):
        print(f" → K={k} 수행 중...")

        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embeddings)

        if len(set(labels)) < 2:
            print(f"   K={k}: 클러스터 1개 → silhouette 계산 불가, skip")
            continue

        sil_score = silhouette_score(embeddings, labels)
        scores[k] = sil_score

        print(f"   K={k}: Silhouette = {sil_score:.4f}")

        if sil_score > best_score:
            best_score = sil_score
            best_k = k

    print(f"\n📌 최적 K = {best_k} (Silhouette={best_score:.4f})")
    return best_k


# ---------------------------
# 시각화
# ---------------------------
def visualize_clusters(embeddings, labels, title="Clustering Result (2D)"):
    pca = PCA(n_components=30)
    reduced = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2, random_state=42, perplexity=40)
    tsne_result = tsne.fit_transform(reduced)

    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='tab10', s=10, alpha=0.7)
    plt.title(title)
    plt.show()


# ---------------------------
# 평가 지표 계산
# ---------------------------
def evaluate_metrics(embeddings, labels):
    unique_labels = np.unique(labels)

    if -1 in unique_labels:
        mask = labels != -1
        embeddings = embeddings[mask]
        labels = labels[mask]
        unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        print("클러스터가 1개 이하라 지표 계산 불가")
        return 0, 0, 0

    # SSE
    sse = 0.0
    for label in unique_labels:
        cluster_points = embeddings[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        sse += np.sum((cluster_points - centroid) ** 2)

    # Silhouette Score
    silhouette = silhouette_score(embeddings, labels)

    # Dunn Index
    inter_dists = []
    intra_dists = []

    for i, c1 in enumerate(unique_labels):
        cluster_i = embeddings[labels == c1]
        for c2 in unique_labels[i+1:]:
            cluster_j = embeddings[labels == c2]
            inter_dists.append(np.min(cdist(cluster_i, cluster_j)))
        intra_dists.append(np.max(cdist(cluster_i, cluster_i)))

    dunn = np.min(inter_dists) / np.max(intra_dists)

    print(f"SSE={sse:.2f}, Silhouette={silhouette:.3f}, Dunn={dunn:.3f}")
    return sse, silhouette, dunn


# ---------------------------
# 메인 실행
# ---------------------------
def main():
    ci = class_image_clustering()
    ci.load_embeddings_npz()
    embeddings = normalize(ci.image_embedding, axis=1)

    results = []

    # --------------------------------------
    # 1️⃣ K-Means (자동 K 탐색)
    # --------------------------------------
    print("\n=== K-MEANS (Optimal K 자동 선택) ===")
    optimal_k = find_best_k(embeddings, k_min=3, k_max=15)

    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    labels_kmeans = kmeans.fit_predict(embeddings)

    ci.print_cluster_counts_and_images(labels_kmeans)
    sse, sil, dunn = evaluate_metrics(embeddings, labels_kmeans)
    visualize_clusters(embeddings, labels_kmeans, f"K-Means (K={optimal_k})")
    results.append(("K-Means", sse, sil, dunn))

    # --------------------------------------
    # 2️⃣ DBSCAN
    # --------------------------------------
    print("\n=== DBSCAN ===")
    eps, min_samples = 0.5, 5
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels_dbscan = db.fit_predict(embeddings)

    ci.print_cluster_counts_and_images(labels_dbscan)
    sse, sil, dunn = evaluate_metrics(embeddings, labels_dbscan)
    visualize_clusters(embeddings, labels_dbscan, f"DBSCAN (eps={eps}, min_samples={min_samples})")
    results.append(("DBSCAN", sse, sil, dunn))

    # --------------------------------------
    # 3️⃣ HDBSCAN
    # --------------------------------------
    print("\n=== HDBSCAN ===")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    labels_hdbscan = clusterer.fit_predict(embeddings)

    ci.print_cluster_counts_and_images(labels_hdbscan)
    sse, sil, dunn = evaluate_metrics(embeddings, labels_hdbscan)
    visualize_clusters(embeddings, labels_hdbscan, "HDBSCAN")
    results.append(("HDBSCAN", sse, sil, dunn))

    # --------------------------------------
    # 4️⃣ 결과 비교표
    # --------------------------------------
    print("\n=== Clustering Performance Comparison ===")
    print(f"{'Algorithm':<10} | {'SSE':<10} | {'Silhouette':<12} | {'Dunn':<10}")
    print("-"*50)

    for name, sse, sil, dunn in results:
        print(f"{name:<10} | {sse:<10.2f} | {sil:<12.3f} | {dunn:<10.3f}")


if __name__ == "__main__":
    main()
