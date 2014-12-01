import numpy as np
import cv2
import os

import constants

train_images = []
train_labels = []

for item in os.listdir(constants.SHARED_TRAIN_PATH):
    if item.endswith(".png"):
        image = cv2.imread(constants.SHARED_TRAIN_PATH + item)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = item[0]
        train_images.append(image)
        train_labels.append(label)

train_images = np.array(train_images).astype(np.float32)
train_labels = np.array(train_labels)

knn = cv2.KNearest()
knn.train(train_images, train_labels)

ret, result, neighbors, dist = knn.find_nearest(train, k=5)
matches = result == test_labels
correct = np.count_nonzero(matches)
accuracy = correct * 100.0 / result.size
print accuracy
