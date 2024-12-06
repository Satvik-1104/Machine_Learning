import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load the IRIS dataset and keep the labels
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target  # This is the label column we will use for labeled initialization

# Function to calculate the Euclidean distance between points
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=1))

# Assign clusters based on closest centroids
def assign_clusters(X, centroids):
    distances = np.array([euclidean_distance(X, centroid) for centroid in centroids])
    return np.argmin(distances, axis=0)

# Update centroids based on mean of assigned points
def update_centroids(X, labels, K):
    centroids = np.zeros((K, X.shape[1]))
    for k in range(K):
        centroids[k] = X[labels == k].mean(axis=0)
    return centroids

# Calculate Sum of Squared Errors (SSE)
def calculate_sse(X, labels, centroids):
    sse = 0
    for k in range(centroids.shape[0]):
        cluster_points = X[labels == k]
        sse += np.sum((cluster_points - centroids[k]) ** 2)
    return sse

# Manual silhouette score calculation
def silhouette_score_manual(X, labels):
    n = len(X)
    A = np.zeros(n)
    B = np.zeros(n)
    for i in range(n):
        same_cluster = labels == labels[i]
        other_clusters = labels != labels[i]
        A[i] = np.mean(np.linalg.norm(X[same_cluster] - X[i], axis=1))
        if np.sum(other_clusters) > 0:
            B[i] = np.min(
                [np.mean(np.linalg.norm(X[labels == k] - X[i], axis=1)) for k in set(labels) if k != labels[i]])
    silhouette_scores = (B - A) / np.maximum(A, B)
    return np.mean(silhouette_scores)

# K-means algorithm with random initialization
def kmeans(X, K, max_iters=100000):
    np.random.seed(42)
    centroids = X[np.random.choice(len(X), K, replace=False)]

    for i in range(max_iters):
        old_centroids = centroids.copy()
        labels = assign_clusters(X, centroids)
        centroids = update_centroids(X, labels, K)
        if np.allclose(centroids, old_centroids):
            break

    sse = calculate_sse(X, labels, centroids)
    silhouette_avg = silhouette_score_manual(X, labels)

    return labels, centroids, sse, silhouette_avg, i + 1

# K-means algorithm with labeled initialization
def kmeans_labeled(X, y, K, max_iters=100):
    unique_labels = np.unique(y)

    if K > len(unique_labels):
        print(
            f"No output: Number of clusters (K={K}) > available classes ({len(unique_labels)})")
        return None, None, None, None, None

    # Select initial centroids from K labeled samples (one from each class)
    labeled_samples = []
    for label in unique_labels[:K]:
        labeled_samples.append(np.where(y == label)[0][0])  # One sample from each class

    centroids = X.iloc[labeled_samples].values  # Using labeled samples to initialize

    for i in range(max_iters):
        old_centroids = centroids.copy()
        labels = assign_clusters(X.values, centroids)
        centroids = update_centroids(X.values, labels, K)
        if np.allclose(centroids, old_centroids):
            break

    sse = calculate_sse(X.values, labels, centroids)
    silhouette_avg = silhouette_score_manual(X.values, labels)

    return labels, centroids, sse, silhouette_avg, i + 1

# Prepare to store results
results = []

# Running both versions of K-means for different K values
k_v = int(input("Enter the value of k(>1): "))
for K in range(2, k_v):
    # Random initialization version
    labels_rand, centroids_rand, sse_rand, silhouette_rand, iters_rand = kmeans(X.values, K)
    results.append({
        "K": K,
        "Initialization": "Random",
        "SSE": sse_rand,
        "Silhouette": silhouette_rand,
        "Iterations": iters_rand
    })

    # Labeled initialization version
    labels_lab, centroids_lab, sse_lab, silhouette_lab, iters_lab = kmeans_labeled(X, y, K)

    if sse_lab is not None:
        results.append({
            "K": K,
            "Initialization": "Labeled",
            "SSE": sse_lab,
            "Silhouette": silhouette_lab,
            "Iterations": iters_lab
        })

# Create a DataFrame from results
results_df = pd.DataFrame(results)

# Printing the Result Dataframe
print()
print(results_df)

# Write results to an Excel file
results_df.to_excel("kmeans_results.xlsx", index=False)

# Plotting the number of iterations for each K value
fig, ax = plt.subplots()
for init in results_df['Initialization'].unique():
    ax.plot(
        results_df[results_df['Initialization'] == init]['K'],
        results_df[results_df['Initialization'] == init]['Iterations'],
        marker='o',
        label=f"{init} Initialization"
    )

ax.set_title("Number of Iterations vs K")
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Number of Iterations")
ax.set_xticks(results_df['K'].unique())
ax.legend()
plt.grid()
plt.show()
