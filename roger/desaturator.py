from PIL import Image, ImageEnhance, ImageOps
import math, os

import constants

# Find the difference between two colors
def colorDifference(c1, c2):
    return math.sqrt(sum([(c1[i]-c2[i])**2 for i in xrange(3)]))

for filename in os.listdir(constants.SHARED_TEST_SRC_PATH):

    if filename[0] == ".": continue # Fucking .DS_Store

    image = Image.open(constants.SHARED_TEST_SRC_PATH + filename)
    image = image.convert("RGB")
    pixdata = image.load()

    print "Computing %d tiles" % (image.size[0] * image.size[1])
    for x in xrange(image.size[0]):
        for y in xrange(image.size[1]):
            difference = colorDifference((20, 20, 20), pixdata[x, y])
            pixdata[x, y] = (255, 255, 255) if difference < 200 else (0, 0, 0)

    image.save(constants.SHARED_TEST_PATH + filename, "JPEG")
