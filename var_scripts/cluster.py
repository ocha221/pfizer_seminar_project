import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

def prepare_clustering_data(df):
    numeric_cols = ['age', 'income', 'quantity', 'spend_amount', 'lat', 'lon']
    boolean_cols = ['is_returning', 'is_q1', 'is_q2', 'is_q3']
    feature_cols = numeric_cols + boolean_cols
    df_features = df[feature_cols].copy()
    df_clean = df_features.dropna()
    
    print(f"Original dataset size: {len(df)}")
    print(f"After removing nulls: {len(df_clean)}")
    print(f"\nFeatures used: {feature_cols}")
    
    return df_clean, feature_cols

def scale_features(df_clean):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    return X_scaled, scaler

def perform_kmeans(X_scaled, df_clean, max_k=10):
    inertias = []
    silhouette_scores = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=919891, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(K_range, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('K-Means Elbow Method')
    ax1.grid(True)
    
    ax2.plot(K_range, silhouette_scores, 'ro-')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('K-Means Silhouette Scores')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('kmeans_elbow_silhouette.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    optimal_k = 3 
    print(f"\nUsing k={optimal_k} clusters")
 
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=919891, n_init=10)
    df_clean['kmeans_cluster'] = kmeans_final.fit_predict(X_scaled)
    
    print(f"Silhouette Score: {silhouette_score(X_scaled, df_clean['kmeans_cluster']):.3f}")
    print(f"Davies-Bouldin Score: {davies_bouldin_score(X_scaled, df_clean['kmeans_cluster']):.3f}")
    
    return df_clean, kmeans_final

def perform_dbscan(X_scaled, df_clean):
    """Perform DBSCAN clustering"""
    print("\n=== DBSCAN CLUSTERING ===")
    
    eps_values = list(np.round(np.linspace(2.25, 4, 151), 3))
    min_samples = 10
    
    """results = []
    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        if n_clusters > 1:
            sil_score = silhouette_score(X_scaled[labels != -1], labels[labels != -1])
        else:
            sil_score = -1
            
        results.append({
            'eps': eps,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette': sil_score
        })
        print(f"eps={eps}: {n_clusters} clusters, {n_noise} noise points, silhouette={sil_score:.3f}")
    """
    
    best_eps = 2.53  # # 3 clusrer silhouette 0.16~
    dbscan_final = DBSCAN(eps=best_eps, min_samples=min_samples)
    df_clean['dbscan_cluster'] = dbscan_final.fit_predict(X_scaled)
    
    print(f"\nUsing eps={best_eps}")
    print(f"Number of clusters: {len(set(df_clean['dbscan_cluster'])) - (1 if -1 in df_clean['dbscan_cluster'].values else 0)}")
    print(f"Noise points: {list(df_clean['dbscan_cluster']).count(-1)}")
    
    return df_clean, dbscan_final

def visualize_clusters(X_scaled, df_clean, feature_cols):
    cluster_summary = df_clean.groupby('kmeans_cluster')[feature_cols].mean()
    print(cluster_summary)
    plt.figure(figsize=(12, 6))
    sns.heatmap(cluster_summary.T, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('K-Means Cluster Characteristics (Mean Values)')
    plt.xlabel('Cluster')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('kmeans_cluster_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    key_features = ['age', 'income', 'spend_amount', 'quantity', 'lat', 'lon']
    
    for idx, feature in enumerate(key_features):
        ax = axes[idx // 3, idx % 3]
        for cluster in sorted(df_clean['kmeans_cluster'].unique()):
            data = df_clean[df_clean['kmeans_cluster'] == cluster][feature]
            ax.hist(data, alpha=0.5, label=f'Cluster {cluster}', bins=20)
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {feature}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('feature_distributions_by_cluster.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    df_clean['kmeans_cluster'].value_counts().sort_index().plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('K-Means Cluster Sizes')
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Count')
    
    dbscan_counts = df_clean['dbscan_cluster'].value_counts().sort_index()
    dbscan_counts.plot(kind='bar', ax=ax2, color='salmon')
    ax2.set_title('DBSCAN Cluster Sizes (−1 = noise)')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('cluster_sizes.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
 
    df = pd.read_csv('filtered_dataset/clustering/no_erroneous_income_filtered_spend.csv')  

    df_clean, feature_cols = prepare_clustering_data(df)
    X_scaled, scaler = scale_features(df_clean)
    
    df_clean, kmeans_model = perform_kmeans(X_scaled, df_clean, max_k=10)
    df_clean, dbscan_model = perform_dbscan(X_scaled, df_clean)
    
    visualize_clusters(X_scaled, df_clean, feature_cols)
    df_clean.to_csv('clustered_data.csv', index=False)
