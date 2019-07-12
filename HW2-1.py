import numpy as np


def creat_img(W, H):
    img = np.array(np.random.random((W, H)) * 100, dtype=np.int64)
    return img


def creat_kernel(m, n):
    kernel = np.array(np.ones((m, n)) / (m*n))
    return kernel


def medianBlur(img, kernel, padding_way='zero'):

    def padding_zero(img):
        img_pad = np.pad(img, ((1, 1), (1, 1)), 'constant')
        return img_pad

    def padding_REPLICA(img):
        img_pad  = np.pad(img, ((1, 1), (1, 1)), 'edge')
        return img_pad

    def cov2(img, f, strde):
        f = kernel
        inw, inh = img.shape
        w, h = f.shape
        outw = int((inw - w) / strde + 1)
        outh = int((inh - h) / strde + 1)
        arr = np.zeros(shape=(outw, outh))
        for g in range(outh):
            for t in range(outw):
                s = 0
                for i in range(w):
                    for j in range(h):
                        s += img[i + g * strde][j + t * strde] * f[i][j]
                arr[g][t] = s
        return arr


    if padding_way == 'zero':
        img = padding_zero(img)
    elif padding_way == 'replica':
        img = padding_REPLICA(img)
    else: print('Wrong Input')

    img = cov2(img, kernel, 1)
    return img

img = creat_img(5, 5)
kernel = creat_kernel(3, 3)
print(img)
print(kernel)
X = medianBlur(img, kernel, 'zero')
print(X)
