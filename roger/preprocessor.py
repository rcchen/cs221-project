from PIL import Image,ImageFilter
import math

import constants

image = Image.open(constants.SHARED_TEST_PATH + "8-0.jpg")
image = image.filter(ImageFilter.FIND_EDGES)
# pixdata = image.load()

#print "Computing %d tiles" % (image.size[0] * image.size[1])
#for x in xrange(image.size[0]):
#    for y in xrange(image.size[1]):
#        difference = colorDifference((50, 30, 40), pixdata[x, y])
#        pixdata[x, y] = (255, 255, 255) if (difference < 40 or (abs(x - 50) < 20 and difference < 100)) else (0, 0, 0)

image.save("image_result.jpg", "JPEG")
