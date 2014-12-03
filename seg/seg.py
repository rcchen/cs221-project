import numpy as np
import scipy
#import pylab
#import mahotas
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import misc

def grayscale(img):
	return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

img = scipy.misc.imread('test.png')
gray = grayscale(img)
plt.imshow(gray, cmap = plt.get_cmap('gray'))
plt.show()