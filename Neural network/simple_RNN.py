import numpy

train_data = [0.1, 0.2, 0.1, 0.4, 0.3, 0.4, 0.6, 0.5, 0.6,
              0.8, 0.7, 0.8, 0.6, 0.5, 0.6, 0.4, 0.3, 0.4, 0.2, 0.1, 0.2]

data = numpy.array(train_data)

lr = 0.1
wb = [
    numpy.array([[1, 1, 1]], dtype=float)*0.01,
    numpy.array([[3, 3, 3], [3, 3, 3], [3, 3, 3]], dtype=float)*0.01,
    numpy.array([[2, 2, 2]], dtype=float)*0.01,
    numpy.array([[3], [3], [3]], dtype=float)*0.01,
    0.01
]


def forward(x, m, wb, grad):
    pre = m
    m = numpy.dot(x, wb[0])+numpy.dot(m, wb[1])+wb[2]
    y = numpy.dot(m, wb[3])+wb[4]
    p = sigmoid(y)
    sp = p*(1-p)
    grad[4] = sp
    grad[3] = sp*pre.T
    grad[2] = sp*wb[3].T
    grad[1] = sp*numpy.dot(pre, wb[3])
    grad[0] = sp*x*wb[3].T
    return m, p, grad


def sigmoid(x):
    return 1/(1+numpy.exp(-x))


def bp(wb, grad):
    for i in range(len(wb)):
        wb[i] = wb[i] - grad[i]*lr
    return wb


for i in range(100000):
    lr = lr * 0.99999
    total_grad = [
        numpy.array([[0, 0, 0]], dtype=float),
        numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float),
        numpy.array([[0, 0, 0]], dtype=float),
        numpy.array([[0], [0], [0]], dtype=float),
        0
    ]
    m = numpy.array([[0, 0, 0]], dtype=float)
    loss = 0
    for j in range(len(train_data)-1):
        grad = [
            numpy.array([[0, 0, 0]], dtype=float),
            numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float),
            numpy.array([[0, 0, 0]], dtype=float),
            numpy.array([[0], [0], [0]], dtype=float),
            0
        ]
        x = data[j]
        r = data[j+1]
        m, p, grad = forward(x, m, wb, grad)
        for i in range(len(wb)):
            grad[i] = grad[i] * (p-r)
            total_grad[i] = total_grad[i] + grad[i]
        loss += 0.5*(r-p)**2
    wb = bp(wb, total_grad)
    print("Loss: ", loss)
print(wb)
