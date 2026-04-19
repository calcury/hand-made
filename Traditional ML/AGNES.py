import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
n = 170


def generate_dbscan_data(n_clusters=3, pts_per_cluster=50, noise_pts=20):
    data = []
    for _ in range(n_clusters):
        center = np.array([np.random.uniform(0, 10), np.random.uniform(0, 10)])
        points = center + np.random.normal(0, 0.5, size=(pts_per_cluster, 2))
        data.extend(points)

    noise = np.random.uniform(0, 10, size=(noise_pts, 2))
    data.extend(noise)

    return np.array(data)


data = generate_dbscan_data(n_clusters=3)
cluster = [i for i in range(n)]
k = 3

distance_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(i+1, n):
        dx = data[i][0] - data[j][0]
        dy = data[i][1] - data[j][1]
        distance_matrix[i][j] = np.sqrt(dx**2 + dy**2)

max_distance = np.max(distance_matrix) + 1e-5
mask_matrix = np.array([[(j >= i) for i in range(n)]for j in range(n)])
mask_matrix = mask_matrix * max_distance
distance_matrix += mask_matrix
num = n


while num > k:
    min_distance = np.min(distance_matrix)
    for i in range(n):
        for j in range(i+1, n):
            if cluster[i] != cluster[j]:
                if distance_matrix[i][j] <= min_distance:
                    num -= 1
                    distance_matrix[i][j] = max_distance
                    index = cluster[j]
                    for m in range(n):
                        if cluster[m] == index:
                            cluster[m] = cluster[i]
            else:
                distance_matrix[i][j] = max_distance


data = np.array(data)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for i in range(len(data)):
    plt.scatter(
        data[i, 0],
        data[i, 1],
        c=colors[cluster[i] % len(colors)],
        s=50,
        alpha=0.5
    )
plt.show()
