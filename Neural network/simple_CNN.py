import random
import math
import numpy as np
import numpy

train_data = [numpy.random.rand(16, 16) for _ in range(24)]

expect = numpy.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])


def generate_train_data():
    train_data = []
    expect = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [1, 0, 1],
                       [0, 1, 0],
                       [0, 0, 1],
                       [1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]])

    expect[6] = [1, 0, 0]
    center = (8, 8)
    size = 4

    # 生成简单的图形图片
    for i in range(12):
        img = np.zeros((16, 16), dtype=np.float32)
        label = expect[i]

        if np.array_equal(label, [1, 0, 0]):
            # 左上角和右下角坐标
            x1, y1 = center[0] - size//2, center[1] - size//2
            x2, y2 = center[0] + size//2, center[1] + size//2
            img[x1:x2+1, y1:y2+1] = 1.0

        # 圆形图片绘制
        elif np.array_equal(label, [0, 1, 0]):
            for x in range(16):
                for y in range(16):
                    # 计算到中心的距离
                    dist = math.sqrt((x - center[0])**2 + (y - center[1])**2)
                    if dist <= size:
                        img[x, y] = 1.0

        # 十字图片绘制
        elif np.array_equal(label, [0, 0, 1]):
            img[center[0], center[1]-size:center[1]+size+1] = 1.0
            img[center[0]-size:center[0]+size+1, center[1]] = 1.0

        # 添加噪声
        noise = np.random.normal(0, 0.05, (16, 16))
        # 限制越界
        img = np.clip(img + noise, 0.0, 1.0)

        train_data.append(img)

    return train_data, expect


train_data, expect = generate_train_data()

kernels = [numpy.random.rand(2*i+3, 2*i+3) for i in range(2)]
bias = [numpy.random.rand(1, 1) for i in range(2)]
# 4个隐藏神经元
w1 = numpy.random.rand(16, 4)
b1 = numpy.zeros((1, 4), dtype=float)
# 3分类
w2 = numpy.random.rand(4, 3)
b2 = numpy.zeros((1, 3), dtype=float)

# ReLU 激活函数


def ReLU(x):
    return numpy.where(x > 0, x, 0)

# ReLU 求导


def ReLUp(x):
    return numpy.where(x > 0, 1, 0)

# softmax 函数


def softmax(x):
    if x.ndim == 1:
        x = x - numpy.max(x)
        return numpy.exp(x) / numpy.sum(numpy.exp(x))
    else:
        x = x - numpy.max(x, axis=1, keepdims=True)
        return numpy.exp(x) / numpy.sum(numpy.exp(x), axis=1, keepdims=True)


def convolution(img, core, bias):
    core_size = core.shape[0]
    padding = (core.shape[0]-1)//2
    img_height, img_width = img.shape
    # 构建卷积核梯度向量
    grad = numpy.zeros(
        (img_height, img_width, core_size, core_size), dtype=float)
    res = img * 0
    img = numpy.pad(img, pad_width=(padding, padding), mode='constant')
    for i in range(img_height):
        for j in range(img_width):
            img_core = img[i:i+core_size, j:j+core_size]
            res[i, j] = sum(sum(img_core*core)) + bias
    # 线性归一化
    d = numpy.max(res) - numpy.min(res)
    for i in range(img_height):
        for j in range(img_width):
            res[i, j] = (res[i, j]-numpy.min(res))/d
            grad[i, j] = img[i:i+core_size, j:j+core_size]/(d+1e-5)
    return res, grad


def pool(img):
    grad_selector = img * 0
    img_height, img_width = img.shape
    if img_height % 2 == 1:
        img = numpy.pad(img, pad_width=((0, 1), (0, 0)), mode='constant')
        img_height += 1
    if img_width % 2 == 1:
        img = numpy.pad(img, pad_width=((0, 0), (0, 1)), mode='constant')
        img_width += 1
    res = numpy.zeros((img_height//2, img_width//2), dtype=float)
    for i in range(img_width//2):
        for j in range(img_height//2):
            flat = img[i*2:i*2+2, j*2:j*2+2].flatten()
            res[i, j] = max(flat)
            for k in range(4):
                if flat[k] == max(flat):
                    grad_selector[i+k//2, j+k % 2] = 1
    return res, grad_selector

# 池化梯度传递


def grad_pool(grad):
    grad_height, grad_width = grad.shape[0:2]
    res = numpy.zeros((grad_height//2, grad_width//2,
                      grad.shape[2], grad.shape[3]), dtype=float)
    for i in range(grad_height//2):
        for j in range(grad_width//2):
            window = grad[i*2:i*2+2, j*2:j*2+2]
            res[i, j] = numpy.sum(window, axis=(0, 1))
    return res

# 前向传播


def forward(img, kernels, bias, w1, b1, w2, b2, r):
    kernels_grad = [numpy.zeros((2*i+3, 2*i+3), dtype=float) for i in range(2)]
    bias_grad = [numpy.zeros((1, 1), dtype=float) for _ in range(2)]
    # 卷积+池化
    img, grad1 = convolution(img, kernels[0], bias[0])
    img, grad_selector1 = pool(img)
    grad1 = grad1*grad_selector1.reshape(16, 16, 1, 1)
    grad1 = grad_pool(grad1)
    img, grad2 = convolution(img, kernels[1], bias[1])
    img, grad_selector2 = pool(img)
    grad1 = grad1*grad_selector2.reshape(8, 8, 1, 1)
    grad1 = grad_pool(grad1)
    grad2 = grad2*grad_selector2.reshape(8, 8, 1, 1)
    grad2 = grad_pool(grad2)
    # 全连接
    v = numpy.array([img.flatten()])
    y1 = numpy.dot(v, w1) + b1
    p1 = ReLU(y1)
    y2 = numpy.dot(p1, w2) + b2
    p2 = softmax(y2)
    ce = p2-r
    # 记录梯度
    grad_w2 = numpy.dot(ce, p2.T)
    grad_b2 = ce
    grad_w1 = numpy.dot(w2, ce.T)
    grad_b1 = numpy.dot(grad_w1, ReLUp(v)).T
    grad_w1 = numpy.dot(grad_w1, ReLUp(v)*v).T
    grad_b1 = sum(grad_b1).reshape(1, 4)
    grad_before = numpy.dot(w2, ce.T)
    grad_before = numpy.dot(grad_before, ReLUp(v)).T*w1
    grad_before = sum(grad_before.T).reshape(4, 4)
    bias_grad[1] = numpy.sum(grad_before, axis=(0, 1))
    bias_grad[0] = numpy.sum(grad_before, axis=(0, 1))*sum(sum(kernels[1]))
    grad1 = grad1.transpose(2, 3, 0, 1)
    grad2 = grad2.transpose(2, 3, 0, 1)
    for i in range(kernels_grad[0].shape[0]):
        for j in range(kernels_grad[0].shape[1]):
            kernels_grad[0][i, j] = sum(sum(grad1[i, j]*grad_before))
    for i in range(kernels_grad[1].shape[0]):
        for j in range(kernels_grad[1].shape[1]):
            kernels_grad[1][i, j] = sum(sum(grad2[i, j]*grad_before))
    kernels_grad[0] = kernels_grad[0]*sum(sum(kernels[1]))
    return (p2, (kernels_grad, bias_grad, grad_w1, grad_b1, grad_w2, grad_b2))


lr = 0.02
for _ in range(1000):
    total_grad_kernels = [
        numpy.zeros((2*i+3, 2*i+3), dtype=float) for i in range(2)
    ]
    total_grad_bias = [numpy.zeros((1, 1), dtype=float) for i in range(2)]
    total_grad_w1 = numpy.zeros((16, 4), dtype=float)
    total_grad_b1 = numpy.zeros((1, 4), dtype=float)
    total_grad_w2 = numpy.zeros((4, 3), dtype=float)
    total_grad_b2 = numpy.zeros((1, 3), dtype=float)
    loss = 0
    for i in range(len(train_data)):
        img = train_data[i]
        r = expect[i]
        p, grads = forward(img, kernels, bias, w1, b1, w2, b2, r)
        ce = p-r
        for j in range(len(total_grad_kernels)):
            total_grad_kernels[j] += grads[0][j]
        for j in range(len(total_grad_bias)):
            total_grad_bias[j] += grads[1][j]
        total_grad_w1 += grads[2]
        total_grad_b1 += grads[3]
        total_grad_w2 += grads[4]
        total_grad_b2 += grads[5]
        loss -= sum(sum(numpy.log(p+1e-3))*r)/3
    for j in range(len(total_grad_kernels)):
        kernels[j] -= total_grad_kernels[j]*lr
    for j in range(len(total_grad_bias)):
        bias[j] -= total_grad_bias[j]*lr
    w1 -= total_grad_w1*lr
    b1 -= total_grad_b1*lr
    w2 -= total_grad_w2*lr
    b2 -= total_grad_b2*lr
    loss = loss/len(train_data)
    if _ % 50 == 0:
        print(f"Epoch {_}, Loss: {loss:.6f}")
    if loss < 0.09:
        break


# 验证集测试
print()
print("="*50)

val_data, val_expect = generate_train_data()

val_loss = 0.0
correct = 0
total = len(val_data)

# 前向推理


def predict(img, kernels, bias, w1, b1, w2, b2):
    # 卷积+池化
    img, _ = convolution(img, kernels[0], bias[0])
    img, _ = pool(img)
    img, _ = convolution(img, kernels[1], bias[1])
    img, _ = pool(img)
    # 全连接
    v = numpy.array([img.flatten()])
    y1 = numpy.dot(v, w1) + b1
    p1 = ReLU(y1)
    y2 = numpy.dot(p1, w2) + b2
    p2 = softmax(y2)
    return p2


for i in range(total):
    img = val_data[i]
    true_label = val_expect[i]
    pred_prob = predict(img, kernels, bias, w1, b1, w2, b2)

    # 计算损失
    val_loss -= np.sum(np.log(pred_prob + 1e-3) * true_label) / 3

    # 计算准确率
    pred_label = np.zeros_like(true_label)
    pred_label[np.argmax(pred_prob)] = 1
    if np.array_equal(pred_label, true_label):
        correct += 1

avg_val_loss = val_loss / total
accuracy = correct / total * 100

print()
print(f"验证集损失: {avg_val_loss:.6f}")
print(f"验证集准确率: {accuracy:.2f}% ({correct}/{total})")
