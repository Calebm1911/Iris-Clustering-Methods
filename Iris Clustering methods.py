#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

# Load and standardize data
iris = load_iris()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(iris.data)

# KMeans clustering
kmeans_scores = {}
for n_clusters in range(2, 6):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_
    kmeans_scores[n_clusters] = silhouette_score(X_scaled, labels)

# Hierarchical clustering
linkage_methods = ["single", "complete", "average"]
hierarchical_scores = {}
for linkage in linkage_methods:
    for n_clusters in range(2, 6):
        model = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage=linkage)
        model.fit(X_scaled)
        labels = model.labels_
        hierarchical_scores[(linkage, n_clusters)] = silhouette_score(X_scaled, labels)

# DBSCAN clustering
db = DBSCAN(eps=0.3, min_samples=2)
db.fit(X_scaled)
labels = db.labels_
dbscan_score = silhouette_score(X_scaled, labels)

# Print results
print("Silhouette Scores:")
print("KMeans:")
for n_clusters, score in kmeans_scores.items():
    print(f"n_clusters = {n_clusters}: {score:.4f}")
print("Hierarchical Clustering:")
if isinstance(hierarchical_scores, dict):  # Corrected variable name
    for (linkage, n_clusters), score in hierarchical_scores.items():  # Corrected variable name
        print(f"Linkage = {linkage}, n_clusters = {n_clusters}: {score:.4f}")
else:
    print(f"Unexpected type for hierarchical_scores: {type(hierarchical_scores)}")

print(f"DBSCAN: {dbscan_score:.4f}")


# In[ ]:




