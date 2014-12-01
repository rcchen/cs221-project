from PIL import Image
import math

import constants

# Find the difference between two colors
def colorDifference(c1, c2):
    return math.sqrt(sum([(c1[i]-c2[i])**2 for i in xrange(3)]))

image = Image.open(constants.SHARED_DEMO_PATH + "image.jpg")
image = image.convert("RGB")
pixdata = image.load()

print "Computing %d tiles" % (image.size[0] * image.size[1])
for x in xrange(image.size[0]):
    for y in xrange(image.size[1]):
        difference = colorDifference((50, 30, 40), pixdata[x, y])
        pixdata[x, y] = (255, 255, 255) if (difference < 40 or (abs(x - 50) < 20 and difference < 100)) else (0, 0, 0)

image.save("image_result.jpg", "JPEG")
