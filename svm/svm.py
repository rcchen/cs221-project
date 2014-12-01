'''
SVM and KNearest digit recognition.
Sample loads a dataset of handwritten digits from '../shared/train/'.
Then it trains and evaluates an SVM.
Following preprocessing is applied to the dataset:
 - Moment-based image deskew (see deskew())
 - Digit images are split into 4 200x200 cells and 16-bin
   histogram of oriented gradients is computed for each
   cell
 - Transform histograms to space with Hellinger metric (see [1] (RootSIFT))
[1] R. Arandjelovic, A. Zisserman
    "Three things everyone should know to improve object retrieval"
    http://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf
Usage:
   python svm.py

Taken from OpenCV source examples, modified for CS 221
'''

import numpy as np
import cv2
import os
import random
import itertools as it
from numpy.linalg import norm

SZ = 100 # size of each digit is SZ x SZ
TRAIN_DIR = '../shared/train/'

def grouper(n, iterable, fillvalue=None):
    '''grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx'''
    args = [iter(iterable)] * n
    return it.izip_longest(fillvalue=fillvalue, *args)

def mosaic(w, imgs):
    '''Make a grid from images. 
    w    -- number of grid columns
    imgs -- images (must have same size and format)
    '''
    imgs = iter(imgs)
    img0 = imgs.next()
    pad = np.zeros_like(img0)
    imgs = it.chain([img0], imgs)
    rows = grouper(w, imgs, pad)
    return np.vstack(map(np.hstack, rows))

def split2d(img, cell_size, flatten=True):
    h, w = img.shape[:2]
    sx, sy = cell_size
    cells = [np.hsplit(row, w//sx) for row in np.vsplit(img, h//sy)]
    cells = np.array(cells)
    if flatten:
        cells = cells.reshape(-1, sy, sx)
    return cells

def load_digits(directory):
    print 'loading "%s" ...' % directory
    digits = list()
    labels = list()
    for filename in os.listdir(directory):
        if not filename.endswith('.png'): continue
        if 'skew' in filename: continue
        digit = cv2.imread(os.path.join(TRAIN_DIR, filename),
            cv2.CV_LOAD_IMAGE_GRAYSCALE)
        digits.append(digit)
        labels.append(int(filename[0]))
    return digits, labels

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

class KNearest(StatModel):
    def __init__(self, k = 3):
        self.k = k
        self.model = cv2.KNearest()

    def train(self, samples, responses):
        self.model = cv2.KNearest()
        self.model.train(samples, responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.find_nearest(samples, self.k)
        return results.ravel()

class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.5):
        self.params = dict( kernel_type = cv2.SVM_RBF,
                            svm_type = cv2.SVM_C_SVC,
                            C = C,
                            gamma = gamma )
        self.model = cv2.SVM()

    def train(self, samples, responses):
        self.model = cv2.SVM()
        self.model.train(samples, responses, params = self.params)

    def predict(self, samples):
        return self.model.predict_all(samples).ravel()


def evaluate_model(model, digits, samples, labels):
    resp = model.predict(samples)
    err = (labels != resp).mean()
    print 'error: %.2f %%' % (err*100)

    confusion = np.zeros((10, 10), np.int32)
    for i, j in zip(labels, resp):
        confusion[i, j] += 1
    print 'confusion matrix:'
    print confusion
    print

def preprocess_simple(digits):
    return np.float32(digits).reshape(-1, SZ*SZ) / 255.0

def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n*ang/(2*np.pi))
        bin_w = SZ / 2
        bin_cells = bin[:bin_w,:bin_w], bin[bin_w:,:bin_w], bin[:bin_w,bin_w:], bin[bin_w:,bin_w:]
        mag_cells = mag[:bin_w,:bin_w], mag[bin_w:,:bin_w], mag[:bin_w,bin_w:], mag[bin_w:,bin_w:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)


if __name__ == '__main__':
    print __doc__

    sorted_digits, sorted_labels = load_digits(TRAIN_DIR)

    print 'preprocessing...'
    # shuffle digits
    indices = range(len(sorted_digits))
    random.shuffle(indices)
    digits = list()
    labels = list()
    for i in indices:
        digits.append(sorted_digits[i])
        labels.append(sorted_labels[i])

    digits2 = map(deskew, digits)
    samples = preprocess_hog(digits2)
    # samples = preprocess_simple(digits2)

    train_n = int(0.9*len(samples))
    # cv2.imshow('test set', mosaic(25, digits[train_n:]))
    # cv2.waitKey(0)
    digits_train, digits_test = np.split(digits2, [train_n])
    samples_train, samples_test = np.split(samples, [train_n])
    labels_train, labels_test = np.split(labels, [train_n])

    print 'training KNearest...'
    model = KNearest(k=4)
    model.train(samples_train, labels_train)
    evaluate_model(model, digits_test, samples_test, labels_test)

    print 'training SVM...'
    model = SVM(C=2.67, gamma=5.383)
    model.train(samples_train, labels_train)
    evaluate_model(model, digits_test, samples_test, labels_test)
    print 'saving SVM as "digits_svm.dat"...'
    model.save('digits_svm.dat')
