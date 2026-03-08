import numpy as np

train_X = [[1, 0, 0, 1, 0],
           [0, 0, 1, 0, 0],
           [1, 1, 0, 1, 0],
           [0, 0, 1, 0, 1],
           [1, 0, 1, 0, 1],
           [1, 0, 1, 0, 1],
           [0, 0, 1, 0, 1],
           [0, 0, 1, 1, 1],
           [1, 0, 0, 0, 1],
           [1, 0, 0, 0, 1]]

train_y = [[1, 1, 0, 0, 1, 1, 0, 0, 1, 0]]

lr = 0.1
data = np.array(train_X)
res = np.array(train_y).T

input_layer = 5
hidden_layer_1 = 9
hidden_layer_2 = 7
output_layer = 1

w = [
    np.random.rand(input_layer, hidden_layer_1),
    np.random.rand(hidden_layer_1, hidden_layer_2),
    np.random.rand(hidden_layer_2, output_layer),
]

b = [
    np.random.rand(1, hidden_layer_1),
    np.random.rand(1, hidden_layer_2),
    np.random.rand(1, output_layer),
]


def fp(data, w, b):
    z = np.dot(data, w) + b
    a = 1/(1+np.exp(-z))
    return z, a


def bp(w, b, z, a, res):
    l = len(train_y)
    Loss = sum(sum((a[2] - res)**2))
    Loss_ = a[2] - res
    a2_ = Loss_*a[2]*(1-a[2])
    a1_ = np.dot(a2_, w[2].T)*a[1]*(1-a[1])
    a0_ = np.dot(a1_, w[1].T)*a[0]*(1-a[0])
    w[2] = w[2] - lr*np.dot(a[1].T, a2_)/l
    w[1] = w[1] - lr*np.dot(a[0].T, a1_)/l
    w[0] = w[0] - lr*np.dot(data.T, a0_)/l
    b[0] = b[0] - lr*a0_/l
    b[1] = b[1] - lr*a1_/l
    b[2] = b[2] - lr*a2_/l
    return Loss


for _ in range(1000):
    z, a = [0]*3, [0]*3
    z[0], a[0] = fp(data, w[0], b[0])
    z[1], a[1] = fp(a[0], w[1], b[1])
    z[2], a[2] = fp(a[1], w[2], b[2])

    Loss = bp(w, b, z, a, res)
    if _ % 100 == 0:
        print(f"Loss: {Loss:.4f}")
