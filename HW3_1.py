import numpy as np
import random

def gen_sample_data():
    w = random.randint(0, 10) + random.random()		# for noise random.random[0, 1)
    b = random.randint(0, 5) + random.random()
    num_samples = 10000
    x_list = []
    y_list = []
    for i in range(num_samples):
        x = random.randint(0, 100) * random.random()
        y = w * x + b + random.random() * random.randint(-1, 1)
        x_list.append(x)
        y_list.append(y)
    return x_list, y_list

def eval_loss(w, b, x_list, gt_y_list):
    error = (gt_y_list - w * x_list + b) ** 2
    loss = 0.5 * sum(error) / len(gt_y_list)
    return loss

def inference(w, b, x):
    pred_y = w * x + b
    return pred_y

def gradient(pred_y, gt_y, x):
    diff = pred_y - gt_y
    dw = diff * x
    db = diff
    return dw, db

def cal_step_gradient(batch_x_list, batch_y_list, w, b, lr):
    prey_y = inference(w, b, batch_x_list)
    dw, db = gradient(prey_y, batch_y_list, batch_x_list)
    dw = sum(dw) / len(batch_x_list)
    db = sum(db) / len(batch_x_list)
    w -= lr * dw
    b -= lr * db
    return w, b

def train(x_list, gt_y_list, batch_size, lr, max_iter):
    w = 0
    b = 0
    for i in range(max_iter):
        batch_idxs = np.random.choice(len(x_list), batch_size)
        batch_x = [x_list[j] for j in batch_idxs]
        batch_x = np.array(batch_x)
        batch_y = [gt_y_list[j] for j in batch_idxs]
        batch_y = np.array(batch_y)
        w, b = cal_step_gradient(batch_x, batch_y, w, b, lr)
        print('w:{0}, b:{1}'.format(w, b))
        x_list, gt_y_list = np.array(x_list), np.array(gt_y_list)
        print('loss is {0}'.format(eval_loss(w, b, x_list, gt_y_list)))

def run():
    x_list, y_list = gen_sample_data()
    lr = 0.0015
    max_iter = 1000
    train(x_list, y_list, 50, lr, max_iter)

if __name__ == '__main__':	# 跑.py的时候，跑main下面的；被导入当模块时，main下面不跑，其他当函数调
    run()
