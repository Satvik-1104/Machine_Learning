import numpy as np
import pandas as pd

np.random.seed(0)

a1 = [6.8, 0.8, 1.2, 2.8, 3.8, 4.4, 4.8, 6.0, 6.2, 7.6, 7.8, 6.6, 8.2, 8.4, 9.0, 9.6]
a2 = [12.6, 9.8, 11.6, 9.6, 9.9, 6.5, 1.1, 19.9, 18.5, 17.4, 12.2, 7.7, 4.5, 6.9, 3.4, 11.1]
dataset = np.column_stack((a1, a2))
dataframe = pd.DataFrame(dataset, columns=['A1', 'A2'])


def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


class KMedoids:
    def __init__(self, clusters, max_itr):
        self.clusters = clusters
        self.max_itr = max_itr
        self.medoids = None
        self.labels = None

    def fit(self, data):
        m = data.shape[0]
        random_indices = np.random.choice(m, self.clusters, replace=False)
        self.medoids = data[random_indices]
        self.labels = np.zeros(m, dtype=int)

        for itr in range(self.max_itr):
            for i in range(m):
                distances = [euclidean_distance(data[i], medoid) for medoid in self.medoids]
                self.labels[i] = np.argmin(distances)

            old_medoids = np.copy(self.medoids)

            for k in range(self.clusters):
                points = data[self.labels == k]
                if len(points) > 0:
                    medoid_costs = [np.sum([euclidean_distance(point, other) for other in points]) for point in points]
                    self.medoids[k] = points[np.argmin(medoid_costs)]

            if np.array_equal(self.medoids, old_medoids):
                print(f"Medoids remain unchanged at iteration {itr + 1}")
                break

        return self.labels

    def cluster_quality(self, data):
        sse = 0
        for k in range(self.clusters):
            points = data[self.labels == k]
            if len(points) > 0:
                for point in points:
                    sse += euclidean_distance(point, self.medoids[k])
        return sse


def silhouette_score(data, labels):
    score = 0.0
    num_clusters = np.max(labels) + 1
    for k in range(num_clusters):
        cluster_points = data[labels == k]
        a = 0
        if len(cluster_points) > 1:
            for point in cluster_points:
                a = np.mean([euclidean_distance(point, other) for other in point if not np.array_equal(point, other)])
                b = np.inf
                for j in range(num_clusters):
                    if j != k:
                        b = min(b, np.mean([euclidean_distance(point, other) for other in cluster_points]))
                score += (b - a) / max(a, b) if a != b else  0
    return score / data.shape[0]


silhouette_scores = []
k_range = range(2, dataset.shape[0])
for k in k_range:
    kmedoids = KMedoids(k, 100)
    kmedoids.fit(dataset)
    score = silhouette_score(dataset, kmedoids.labels)
    silhouette_scores.append(score)

opt_k = k_range[np.argmax(silhouette_scores)]
print(f"Optimal K: {opt_k}")

max_itr_list = [10, 50, 100, 200]
for max_iter in max_itr_list:
    kmedoids = KMedoids(opt_k, max_iter)
    final_labels = kmedoids.fit(dataset)
    medoids = kmedoids.medoids

kmedoids = KMedoids(opt_k, 100)
final_labels = kmedoids.fit(dataset)
medoids = kmedoids.medoids
silhouette_score_final = silhouette_score(dataset, final_labels)

print(f"\nSilhouette Score for optimal clusters: {silhouette_score_final:.4f}")
for i in range(len(medoids)):
    print(f"M{i}: ({medoids[i][0]:.2f}, {medoids[i][1]:.2f})")

for i in range(len(final_labels)):
    print(f"({dataset[i][0]}, {dataset[i][1]}):  Cluster {final_labels[i] + 1}")
