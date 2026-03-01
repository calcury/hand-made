import numpy as np

train_data = [[1, 0, 0, 1, 0],
              [0, 0, 1, 1, 0],
              [1, 1, 0, 1, 0],
              [1, 0, 0, 0, 1],
              [0, 0, 1, 0, 1],
              [1, 1, 0, 0, 1]]

predict = [[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1],
           [1, 0, 0],
           [0, 1, 0],
           [0, 0, 1]]

exp = 0.05
data = np.array(train_data)
w1 = np.ones((5, 8), dtype=float)
b1 = np.ones((1, 8), dtype=float)
w1 = w1/8
b1 = b1/8
w2 = np.zeros((8, 3), dtype=float)
b2 = np.zeros((1, 3), dtype=float)


def softmax(x):
    if x.ndim == 1:
        x = x - np.max(x)
        return np.exp(x) / np.sum(np.exp(x))
    else:
        x = x - np.max(x, axis=1, keepdims=True)
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def ReLU(x):
    return np.where(x > 0, x, 0)


def fp(data, w1, b1, w2, b2):
    temp1 = np.dot(data, w1) + b1
    res1 = ReLU(temp1)
    temp2 = np.dot(res1, w2) + b2
    res2 = softmax(temp2)
    return [res1, res2, temp1, temp2]


def ReLUprime(x):
    return np.where(x >= 0, 1, 0)


def bp(data, res1, res2, predict, w1, b1, w2, b2, t1, t2):
    n = data.shape[0]
    # w2/b2
    rows, cols = res1.shape
    categories = len(predict[0])
    for k in range(categories):
        for i in range(cols):
            loss_w = 0
            for j in range(rows):
                x = res1[j, i]
                loss_w += x*(res2[j][k]-predict[j][k])
            w2[i][k] = w2[i][k] - loss_w/n*exp
        loss_b = 0
        for i in range(rows):
            loss_b += res2[i][k]-predict[i][k]
        b2[0][k] = b2[0][k] - loss_b/n*exp
    r = rows
    # w1/b1
    rows, cols = data.shape
    loss = res2-predict
    loss = np.dot(loss, w2.T)
    loss = loss*ReLUprime(t1)
    b1 = b1 - sum(loss)/n*exp
    loss = np.dot(loss.T, data).T
    w1 = w1 - loss/n*exp
    print("Loss:", -sum(sum(predict*np.log(res2))))
    if not -sum(sum(predict*np.log(res2))):
        return 0
    return w1, b1, w2, b2


for i in range(10000):
    res1, res2, t1, t2 = fp(data, w1, b1, w2, b2)
    w1, b1, w2, b2 = bp(data, res1, res2, predict, w1, b1, w2, b2, t1, t2)

test1 = [1, 1, 0, 1, 0]
test2 = [1, 0, 0, 1, 0]
print(fp(test1, w1, b1, w2, b2)[1])
print(fp(test2, w1, b1, w2, b2)[1])
