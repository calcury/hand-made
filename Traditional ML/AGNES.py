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
k = 10

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

flat_idx = np.argsort(distance_matrix.ravel())
row_idx, col_idx = np.unravel_index(flat_idx, distance_matrix.shape)
sorted_positions = list(zip(row_idx, col_idx))

num = n


class DSU:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)

        if root_i != root_j:
            self.parent[root_i] = root_j
            return True
        return False


dsu = DSU(n)
num = n

for pos in sorted_positions:
    i, j = pos
    if dsu.find(i) != dsu.find(j):
        dsu.union(i, j)
        num -= 1

    if num <= k:
        break

cluster = [dsu.find(i) for i in range(n)]
data = np.array(data)

# Graph
unique_roots = list(set(cluster))
root_map = {root: i for i, root in enumerate(unique_roots)}
cluster = [root_map[c] for c in cluster]

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
