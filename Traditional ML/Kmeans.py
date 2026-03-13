import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

n = 200
k = 3
data = [[np.random.rand(), np.random.rand()] for _ in range(n)]
category = [0]*n
center = [[np.random.rand(), np.random.rand()] for _ in range(k)]

epochs = 10
for _ in range(epochs):
    for i in range(n):
        d = [0]*k
        for j in range(k):
            d[j] = ((data[i][0]-center[j][0])**2 +
                    (data[i][1]-center[j][1])**2)
        for j in range(k):
            if d[j] == min(d):
                category[i] = j
    c = [[0, 0, 0] for _ in range(k)]
    for i in range(n):
        c[category[i]][0] += data[i][0]
        c[category[i]][1] += data[i][1]
        c[category[i]][2] += 1
    for i in range(k):
        center[i][0] = c[i][0] / (c[i][2]+1e-8)
        center[i][1] = c[i][1] / (c[i][2]+1e-8)

data = np.array(data)
center = np.array(center)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for i in range(len(data)):
    plt.scatter(
        data[i, 0],
        data[i, 1],
        c=colors[category[i] % len(colors)],
        s=50,
        alpha=0.5
    )

plt.scatter(
    center[:, 0],
    center[:, 1],
    c=colors[:k],
    marker='s',
    edgecolors='black',
)
plt.show()
