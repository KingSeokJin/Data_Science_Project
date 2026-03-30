import os
import random
import shutil


import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

import pickle

from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import math


from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist


class class_image_clustering:
    def __init__(self):
        self.tar_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'image_samples')
        self.num_images = 3000
        self.embedding_size = 2048
        
        self.idx2filename, self.filename2idx = None, None
        self.image_embedding = None
        
        self.embedding_file_npz = "image_embedding.npz"

        
    def load_embeddings_npz(self):

        data = np.load(self.embedding_file_npz, allow_pickle=True)
        filenames = data['filenames']
        self.idx2filename = {idx: fn for idx, fn in enumerate(filenames)}
        self.filename2idx = {fn: idx for idx, fn in self.idx2filename.items()}
        self.image_embedding = data['image_embedding']
        
        print(f"임베딩 로드 완료: {len(self.image_embedding)}개") 
        
 
    def topk_similar_images(self, image_file, k):
    
        image_idx = self.filename2idx[image_file]
        query_image_vector = self.image_embedding[image_idx]
    
        similarities = cosine_similarity([query_image_vector], self.image_embedding)[0]
        print(f"similarity shape = {similarities.shape}")
    
        topk_indices = np.argsort(similarities)[::-1][:k]
        print(f"topk_indices={topk_indices}\n")
    
        topk_image_files = [os.path.join(self.tar_dir, self.idx2filename[idx]) for idx in topk_indices]
        print(f"topk_image_files={topk_image_files}\n")
        
        return topk_image_files

    
    def show_images(self, image_filenames):
        num_images = len(image_filenames)
        
        img_width, img_height = 300, 300  
        
        fig, axes = plt.subplots(nrows=num_images, ncols=1, figsize=(img_width / 100, img_height * num_images / 100), dpi=100)
    
        if num_images == 1:
            axes = [axes]  # 이미지가 하나인 경우에도 배열로 취급
    
        for ax, image_file in zip(axes, image_filenames):
            img = Image.open(image_file)
            ax.imshow(img)
            ax.axis('off')  # 축을 숨김
        
        plt.tight_layout()
        plt.show()

    def perform_kmeans(self, n_clusters=10):

        normalized_embeddings = normalize(self.image_embedding, axis=1)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(normalized_embeddings)
        
        labels = kmeans.labels_
        return normalized_embeddings, labels

    def print_cluster_counts(self, labels):
        unique, counts = np.unique(labels, return_counts=True)
        cluster_counts = dict(zip(unique, counts))
        
        print("클러스터별 이미지 개수:")
        for cluster, count in cluster_counts.items():
            print(f"클러스터 {cluster}: {count}장")
            
    def print_cluster_counts_and_images(self, labels):
        unique, counts = np.unique(labels, return_counts=True)
        cluster_counts = dict(zip(unique, counts))

        print("클러스터별 이미지 개수:")
        for cluster, count in cluster_counts.items():
            print(f"클러스터 {cluster}: {count}장")

            # 해당 클러스터의 이미지 인덱스 찾기
            cluster_indices = np.where(labels == cluster)[0]

            # 클러스터 중심점 계산 (각 클러스터 내 임베딩 벡터들의 평균)
            cluster_embeddings = self.image_embedding[cluster_indices]
            centroid = np.mean(cluster_embeddings, axis=0)

            # 클러스터 내 이미지들과 중심점 간의 거리 계산
            distances = [np.linalg.norm(self.image_embedding[idx] - centroid) for idx in cluster_indices]
            
            # 거리가 가까운 순으로 정렬하여 가장 가까운 6개 선택
            closest_indices = [cluster_indices[i] for i in np.argsort(distances)[:6]]

            fig, axs = plt.subplots(1, 6, figsize=(15, 3))  
            fig.suptitle(f"cluster {cluster} images")
            
            for i, idx in enumerate(closest_indices):
                img_path = os.path.join(self.tar_dir, self.idx2filename.get(idx, ''))
                
                if os.path.exists(img_path):  
                    try:
                        img = Image.open(img_path)
                        img = img.resize((300, 300)) 
                        axs[i].imshow(img)
                    except Exception as e:
                        print(f"이미지 로드 실패: {img_path}, 오류: {e}")
                        axs[i].imshow(np.zeros((300, 300, 3), dtype=np.uint8))  # 빈 이미지
                else:
                    print(f"파일 없음: {img_path}")
                    axs[i].imshow(np.zeros((300, 300, 3), dtype=np.uint8))  # 빈 이미지
                axs[i].axis('off')  # 축 숨기기
            
            for j in range(len(closest_indices), 6):
                axs[j].imshow(np.zeros((300, 300, 3), dtype=np.uint8))  # 빈 이미지
                axs[j].axis('off')
            
            plt.show()



    def evaluate(self, embeddings, labels):

        silhouette_avg = silhouette_score(embeddings, labels)
        print(f"Silhouette Score: {silhouette_avg:.4f}")

        sse = self.calculate_sse(embeddings, labels)
        print(f"SSE (Sum of Squared Errors): {sse:.4f}")

        dunn_index = self.calculate_dunn_index(embeddings, labels)
        print(f"Dunn Index: {dunn_index:.4f}")

    def calculate_sse(self, embeddings, labels):
        sse = 0.0
        for label in np.unique(labels):
            cluster_points = embeddings[labels == label]
            centroid = np.mean(cluster_points, axis=0)
            sse += np.sum((cluster_points - centroid) ** 2)
        return sse


    def calculate_dunn_index(self, embeddings, labels):
        unique_labels = np.unique(labels)
        inter_cluster_distances = []
        intra_cluster_distances = []

        # 클러스터 간 최소 거리 계산 (inter-cluster distance)
        for i, label_i in enumerate(unique_labels):
            cluster_i = embeddings[labels == label_i]
            for label_j in unique_labels[i + 1:]:
                cluster_j = embeddings[labels == label_j]
                distances = cdist(cluster_i, cluster_j)  # 클러스터 간 모든 포인트 쌍의 거리
                inter_cluster_distances.append(np.min(distances))  # 최소 거리 선택

        # 클러스터 내 최대 거리 계산 (intra-cluster distance)
        for label in unique_labels:
            cluster_points = embeddings[labels == label]
            if len(cluster_points) > 1:  # 클러스터 내 포인트가 2개 이상일 때만 거리 계산
                distances = cdist(cluster_points, cluster_points)
                intra_cluster_distances.append(np.max(distances))

        # Dunn Index = (최소 클러스터 간 거리) / (최대 클러스터 내 거리)
        if intra_cluster_distances:  # intra-cluster distance가 존재할 경우에만 계산
            dunn_index = np.min(inter_cluster_distances) / np.max(intra_cluster_distances)
            return dunn_index
        else:
            return 0  # 클러스터 내 거리가 없으면 0 반환

    def clustering_metrics(self):
        silhouette_scores = []
        sse_scores = []
        dunn_indices = []  
        
        k_values = list(range(3,31))
        
        for k in k_values:
            embeddings, labels = self.perform_kmeans(n_clusters=k)
            print(f"\nk={k} clustering performance:")
            silhouette = silhouette_score(embeddings, labels)
            sse = self.calculate_sse(embeddings, labels)
            dunn_index = self.calculate_dunn_index(embeddings, labels)
            print(f"silhouette={silhouette}")
            print(f"sse={sse}")
            print(f"dunn_index={dunn_index}")            
    
            silhouette_scores.append(silhouette)
            sse_scores.append(sse)          
            dunn_indices.append(dunn_index)

        # Silhouette Score Plot
        plt.figure(figsize=(12, 4))
        plt.plot(k_values, silhouette_scores, marker='o', label='Silhouette Score')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score vs. Number of Clusters')
        plt.xticks(k_values)
        plt.grid(True)
        plt.legend()
        plt.show()
        
        # SSE Plot
        plt.figure(figsize=(12, 4))
        plt.plot(k_values, sse_scores, marker='o', label='SSE')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('SSE (Sum of Squared Errors)')
        plt.title('SSE vs. Number of Clusters')
        plt.xticks(k_values)
        plt.grid(True)
        plt.legend()
        plt.show()
        
        
        # Dunn Index Plot
        plt.figure(figsize=(12, 4))
        plt.plot(k_values, dunn_indices, marker='o', label='Dunn Index')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Dunn Index')
        plt.title('Dunn Index vs. Number of Clusters')
        plt.xticks(k_values)
        plt.grid(True)
        plt.legend()
        plt.show()            


def main():
    ci = class_image_clustering()
    ci.load_embeddings_npz()
    
    embeddings, labels = ci.perform_kmeans(n_clusters=10)
    ci.print_cluster_counts_and_images(labels)    
    
    #ci.clustering_metrics()


if __name__ == '__main__':

    main()
    
    