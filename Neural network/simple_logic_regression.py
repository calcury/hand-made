import numpy as np

train_data = [[1, 0, 0, 1, 0],
              [0, 0, 1, 0, 0],
              [1, 1, 0, 1, 0],
              [1, 0, 1, 0, 1]]

predicted_value = [1, 1, 0, 0]

lr = 0.02
data = np.array(train_data)

w = np.zeros((5, 1), dtype=float)
b = 0


def sigmoid(x):
    return 1/(1+np.exp(-x))


def fp(data, w, b):
    p = np.dot(data, w) + b
    res = sigmoid(p)
    return res, p


def bp(data, res, predict, w, p, b):
    rows, cols = data.shape
    b_loss = 0
    for i in range(cols):
        w_loss = 0
        for j in range(rows):
            x = data[j, i]
            w_loss += x*np.exp(-p[j])/(1+np.exp(-p[j]))**2*(res[j]-predict[j])
        w[i] = w[i] - w_loss/rows*lr
    loss = w_loss.item()
    print("Loss:", loss/rows)
    for i in range(rows):
        b_loss += np.exp(-p[i])/(1+np.exp(-p[i]))**2*(res[i]-predict[i])
    b = b - b_loss/rows*lr
    return w, b


for i in range(1000):
    res, p = fp(data, w, b)
    w, b = bp(data, res, predicted_value, w, p, b)

test1 = [0, 0, 1, 0, 0]
test2 = [1, 0, 1, 0, 1]
print(sigmoid(np.dot(test1, w)+b))
print(sigmoid(np.dot(test2, w)+b))
