import numpy as np

def gen_sample_data():
    n = 1000  # number of points to create
    data = []
    for i in range(n):  # give n=1000 date diferent value
        [r0, r1] = np.random.standard_normal(2)  # create a list whitch has two elements
        myClass = np.random.uniform(0, 1)  # used to classify
        if (myClass < 0.33):
            fFlyer = 6000 * r0 + 7000
            tats = 10 + 3 * r1 + 2 * r0
            data.append([fFlyer, tats, 0])

        elif (myClass > 0.66):
            fFlyer = 1000 * r0 + 3500
            tats = 10 + 2.0 * r1
            data.append([fFlyer, tats, 1])

    data = np.array(data)
    return data

def sampling_data():
    data = gen_sample_data()
    dataMAT = data[:, 0:-1]
    labelMAT= data[:, -1]
    dataMAT = np.insert(dataMAT, 0, 1, axis=1)
    return dataMAT, labelMAT

def sigmoid(x):
    return 2 * 1.0/(1+np.exp(-2 * x)) - 1

def loss_funtion(dataMat, classLabels, weights):
    m, n = np.shape(dataMat)
    loss = 0.0
    for i in range(m):
        h = sigmoid(dataMat[i] * weights)
        loss -= classLabels[i] * np.log(h) + (1 - classLabels[i]) * np.log(1-h)
    loss /= m
    return loss

def grad_descent(dataMatIn, classLabels, weights, lr):
    dataMatrix = dataMatIn
    labelMat = classLabels
    m, n = np.shape(dataMatrix)
    avg_weights = 0
    alpha = lr

    for i in range(m):
        h_theta_x = sigmoid(sum(dataMatrix[i] * weights))
        e = h_theta_x - labelMat[i]
        avg_weights = avg_weights - alpha * dataMatrix[i] * e

    avg_weights /= m

    return avg_weights

def train(dataMAT, labelMAT, batch_size, lr, max_iter):
    m, n = np.shape(dataMAT)
    weights = np.ones(n)
    for i in range(max_iter):
        batch_index = np.random.choice(m, batch_size)
        batch_x = [dataMAT[j] for j in batch_index]
        batch_y = [labelMAT[j] for j in batch_index]
        weights = grad_descent(batch_x, batch_y, weights, lr)
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        weights = np.array(weights)
        loss = loss_funtion(batch_x, batch_y, weights)
        print('w:{0}'.format(weights))
        print('l:{0}'.format(loss))

def run():
    x_list, y_list = sampling_data()
    lr = 0.01
    max_iter = 500
    train(x_list, y_list, 100, lr, max_iter)

if __name__ == '__main__':
    run()

