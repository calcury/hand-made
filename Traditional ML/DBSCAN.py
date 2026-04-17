import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)

EPS = 1
MINPTS = 5

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
level = [0]*n
center = [-1]*n


def distance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return np.sqrt(dx**2 + dy**2)


def dbscan(data, eps, minpts):
    for i in range(n):
        c = 0
        for j in range(n):
            if distance(data[i], data[j]) < eps:
                c += 1
        if c >= minpts:
            level[i] = 2
    for i in range(n):
        for j in range(n):
            if distance(data[i], data[j]) < eps and level[i] == 2 and level[j] == 0:
                level[j] = 1
    group = 0
    for i in range(n):
        if center[i] == -1 and level[i] == 2:
            cluster(data, eps, i, group)
            group += 1
    return center


def cluster(data, eps, i, group):
    center[i] = group
    q = [i]
    while q:
        k = q.pop(0)
        if level[k] != 2:
            continue
        for j in range(n):
            if distance(data[k], data[j]) < eps and center[j] == -1:
                center[j] = group
                q.append(j)


center = dbscan(data, EPS, MINPTS)
data = np.array(data)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for i in range(len(data)):
    plt.scatter(
        data[i, 0],
        data[i, 1],
        c=colors[center[i] % len(colors)],
        s=50,
        alpha=0.5
    )
plt.show()
