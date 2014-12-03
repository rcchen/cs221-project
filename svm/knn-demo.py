'''
SVM, KNearest, and Random Forest digit recognition.

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

import json
import numpy as np
import cv2
import os
import random
import sys
import itertools as it
from numpy.linalg import norm

SZ = 200 # size of each digit is SZ x SZ
TRAIN_DIR = '../shared/train/'
TEST_DIR = '../shared/test/'
DEMO_DIR = '../shared/demo/'
USE_TEST_DIR = True

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

def load_test_digits(directory):
    print 'loading "%s" ...' % directory
    digits = list()
    labels = list()
    
    test_clusters = dict()
    for filename in os.listdir(directory):
        print filename
        if filename[0] == ".": continue
        params = filename.split("_")
        if params[0] not in test_clusters:
            test_clusters[params[0]] = dict()
        if params[1] not in test_clusters[params[0]]:
            test_clusters[params[0]][params[1]] = list()
        test_clusters[params[0]][params[1]].append(params[2])

    digits = list()
    labels = list()
    images_sorted = sorted(test_clusters.keys())
    for image in images_sorted:
        clusters_sorted = sorted(test_clusters[image].keys())
        digit_image = list()
        label_image = list()
        for cluster in clusters_sorted:
            indices = sorted(test_clusters[image][cluster])
            digit_cluster = list()
            label_cluster = list()
            for index in indices:
                filename = "_".join((image, cluster, index))
                label = 0
                digit = cv2.imread(os.path.join(directory, filename),
                    cv2.CV_LOAD_IMAGE_GRAYSCALE)
                digit_cluster.append(digit)
                label_cluster.append(label)
            digit_image.append(digit_cluster)
            label_image.append(label_cluster)
        digits.append(digit_image)
        labels.append(label_image)

    return digits, labels



def load_digits(directory):
    print 'loading "%s" ...' % directory
    digits = list()
    labels = list()
    for filename in os.listdir(directory):
        if filename[0] == ".": continue
        # if not filename.endswith('.png'): continue
        # if 'skew' in filename: continue
        digit = cv2.imread(os.path.join(directory, filename),
            cv2.CV_LOAD_IMAGE_GRAYSCALE)
        if digit is None:
            print filename
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
        self.model.train(samples, np.array(responses))

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.find_nearest(samples, self.k)
        return results.ravel()

# Play around with the kernel type
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

class RTree(StatModel):

  def train(self, samples, responses):
    self.model = cv2.RTrees()
    self.model.train(samples, cv2.CV_ROW_SAMPLE, np.array(responses))

  def predict(self, samples):
    return [self.model.predict(sample) for sample in samples]

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


def load_raw_data(use_test_dir):
    if use_test_dir:
        digits_train, labels_train = load_digits(TRAIN_DIR)
        digits_test, labels_test = load_test_digits(DEMO_DIR)
        return digits_train, np.array(labels_train), digits_test, labels_test
    else:
        sorted_digits, sorted_labels = load_digits(TRAIN_DIR)
        # shuffle digits
        rand = np.random.RandomState(321)
        indices = range(len(sorted_digits))
        random.shuffle(indices)
        digits = list()
        labels = list()
        for i in indices:
            digits.append(sorted_digits[i])
            labels.append(sorted_labels[i])
        train_n = int(0.9*len(sorted_digits))
        digits_train, digits_test = np.split(digits, [train_n])
        labels_train, labels_test = np.split(labels, [train_n])
        return digits_train, labels_train, digits_test, labels_test

if __name__ == '__main__':
    # print __doc__

    # Make sure labels go through numpy.array
    players_json = '{"11":"Jordan, Dontonio","32":"Alfieri, Joey","91":"Anderson, Henry","48":"Anderson, Kevin","93":"Austin, Brendon","21":"Harris, Ronnie","98":"Bright, David","73":"Burkett, Jesse","17":"Tarpley, A.J.","89":"Cajuste, Devon","95":"Callihan, Lance","25":"Carter, Alex","57":"Caspers, Johnny","45":"Chandler, Calvin","10":"Hoffpauir, Zach","80":"Cotton, Eric","81":"Crane, Conner","5":"Whitfield, Kodi","66":"Phillips, Harrison","3":"Rector, Michael","71":"Fanaika, Brandon","44":"Flacco, John","28":"Franklin, Denzel","46":"Gaertner, Ryan","51":"Garnett, Joshua","69":"Grace, Jim","75":"Watkins, Jordan","82":"Harrell, Chris","97":"Hayes, Anthony","40":"Hemschoot, Joe","60":"Hinds, Lucas","8":"Richards, Jordan","31":"Holder, Alijah","84":"Hooper, Austin","86":"Veach, Lane","41":"Johnson, Addison","59":"Jones, Craig","34":"Ukropina, Conrad","99":"Kaumatule, Luke","68":"Keller, C.J.","38":"Pippens, Ra\'Chard","29":"Lloyd, Dallas","55":"Lohn, Nate","43":"Lueders, Blake","2":"Lyons, Wayne","4":"Martinez, Blake","35":"Marx, Daniel","27":"McCaffrey, Christian","42":"McFadden, Pat","67":"Miller, Reed","7":"Shittu, Aziz","23":"Murphy, Alameen","78":"Murphy, Kyle","20":"Okereke, Bobby","22":"Wright, Remound","79":"Yazdi, Alex","6":"Thomas, Taijuan","49":"Palma, Kevin","58":"Parry, David","70":"Peat, Andrus","15":"Perez, Jordan","96":"Plantaric, Eddie","87":"Pratt, Jordan","63":"Reihner, Kevin","14":"Rhyne, Ben","47":"Yules, Sam","53":"Rotto, Torsten","72":"Salem, J.B.","26":"Sanders, Barry","9":"Vaughters, James","30":"Seale, Ricky","50":"Shober, Sam","52":"Shuler, Graham","19":"Williamson, Jordan","24":"Skov, Patrick","13":"Stallworth, Rollins","88":"Taboada, Greg","90":"Thomas, Solomon","18":"Trojan, Jeff","62":"Tubbs, Austin","77":"Tucker, Casey","33":"Tyler, Mike","36":"Ward, Lee","39":"Young, Kelsey"}'
    players = json.loads(players_json)

    digits_train, labels_train, digits_tests, labels_tests = load_raw_data(USE_TEST_DIR)

    print "Preprocessing and training with KNearest..."
    digits_train = map(deskew, digits_train)
    samples_train = preprocess_simple(digits_train)
    model = KNearest(k=3)
    model.train(samples_train, labels_train)

    for i, digit_test in enumerate(digits_tests):
        for j, cluster_tests in enumerate(digit_test):
            sys.stdout.write("Cluster " + str(i) + "." + str(j) + ": ")
            player = ""
            for cluster_test in cluster_tests:
                # print type(cluster_test)
                digit = map(deskew, [cluster_test])
                # print type(digit)
                sample = preprocess_simple(digit)
                # print type(sample)
                digit_result = str(int(model.predict(sample)[0]))
                sys.stdout.write(digit_result)
                player += digit_result
            sys.stdout.write(" (" + players[player] + ")")
            sys.stdout.write("\n")
            sys.stdout.flush()


    # print 'preprocessing...'
    # digits_train = map(deskew, digits_train)
    # digits_test = map(deskew, digits_test)

    # print 'training KNearest...'
    # samples_train = preprocess_simple(digits_train)
    # samples_test = preprocess_simple(digits_test)
    # model = KNearest(k=3)
    # model.train(samples_train, labels_train)
    # evaluate_model(model, digits_test, samples_test, labels_test)

    # print 'training Random Forest...'
    # samples_train = preprocess_hog(digits_train)
    # samples_test = preprocess_hog(digits_test)
    # model = RTree()
    # model.train(samples_train, labels_train)
    # evaluate_model(model, digits_test, samples_test, labels_test)

    # print 'training SVM...'
    # samples_train = preprocess_hog(digits_train)
    # samples_test = preprocess_hog(digits_test)
    # model = SVM(C=2.67, gamma=5.383)
    # model.train(samples_train, labels_train)
    # evaluate_model(model, digits_test, samples_test, labels_test)
    # print 'saving SVM as "digits_svm.dat"...'
    # model.save('digits_svm.dat')
