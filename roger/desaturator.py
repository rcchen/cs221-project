from PIL import Image, ImageEnhance, ImageOps, ImageDraw
import math, os

import constants

# Find the difference between two colors
def colorDifference(c1, c2):
    return math.sqrt(sum([(c1[i]-c2[i])**2 for i in xrange(3)]))    

# Taken from http://stackoverflow.com/questions/4321290/how-do-i-make-pil-take-into-account-the-shortest-side-when-creating-a-thumbnail
def thumbnail(img, size=150):

    from math import floor
    from PIL import Image

    img = img.copy()
    width, height = img.size
    if width == height:
        img.thumbnail((size, size))
    elif height > width:
        ratio = float(width) / float(height)
        newwidth = ratio * size
        img = img.resize((int(floor(newwidth)), size))
    elif width > height:
        ratio = float(height) / float(width)
        newheight = ratio * size
        img = img.resize((size, int(floor(newheight))))
    return img

# Returns only the largest contiguous black region of an image
def largestContiguousRegion(image):

    largest = -1
    source_x = 0
    source_y = 0
    largest_flood = set()
    largest_flood_lower_x = 0
    largest_flood_lower_y = 0
    largest_flood_upper_x = 0
    largest_flood_upper_y = 0

    def pixelInImage(pixel):
        return pixel[0] < image.size[0] and pixel[0] > 0 \
            and pixel[1] < image.size[1] and pixel[1] > 0

    pixdata = image.load()

    flooded = set()
    for x in xrange(image.size[0]):
        for y in xrange(image.size[1]):
            if (x, y) in flooded: continue
            count = 0
            visited = set()
            potential_flood = set()
            flood_lower_x = image.size[0]
            flood_lower_y = image.size[1]
            flood_upper_x = -1
            flood_upper_y = -1
            queue = [(x, y)]
            while len(queue) > 0:
                current = queue.pop(0)
                flooded.add(current)
                if pixelInImage(current) and pixdata[current[0], current[1]] == (0, 0, 0) and current not in visited:
                    if current[0] < flood_lower_x:
                        flood_lower_x = current[0]
                    if current[0] > flood_upper_x:
                        flood_upper_x = current[0]
                    if current[1] < flood_lower_y:
                        flood_lower_y = current[1]
                    if current[1] > flood_upper_y:
                        flood_upper_y = current[1]
                    count += 1
                    visited.add(current)
                    potential_flood.add(current)
                    queue.append((current[0] + 1, current[1]))
                    queue.append((current[0] - 1, current[1]))
                    queue.append((current[0], current[1] + 1))
                    queue.append((current[0], current[1] - 1))
                    # for i in [-1, 0, 1]:
                    #     for j in [-1, 0, 1]:
                    #         if not (i == 0 and j == 0):
                    #             queue.append((current[0] + i, current[1] + j))
            if count > largest:
                largest = count
                largest_flood = potential_flood.copy()
                source_x = x
                source_y = y
                largest_flood_lower_x = flood_lower_x
                largest_flood_lower_y = flood_lower_y
                largest_flood_upper_x = flood_upper_x
                largest_flood_upper_y = flood_upper_y

    # Create a new image that is completely flood based
    flooded = Image.new("1", (image.size[0], image.size[1]), 1)
    flooded = flooded.convert("P", dither=None)
    flooded_pixdata = flooded.load()
    for pixel in largest_flood:
        flooded_pixdata[pixel[0], pixel[1]] = 0
    print flooded.size
    flooded = flooded.crop((largest_flood_lower_x, largest_flood_lower_y, largest_flood_upper_x, largest_flood_upper_y))
    print flooded.size

    result = Image.new("1", (200, 200), 1)
    result = result.convert("P", dither=None)
    result_thumbnail = thumbnail(flooded, 170)
    result.paste(result_thumbnail, (100 - result_thumbnail.size[0] / 2, 15))

    return result

for filename in os.listdir(constants.SHARED_TEST_SRC_PATH):

    if filename[0] == ".": continue # Fucking .DS_Store

    # Part 1, color isolation
    image = Image.open(constants.SHARED_TEST_SRC_PATH + filename)
    image = image.convert("RGB")
    pixdata = image.load()

    print "Computing %d tiles" % (image.size[0] * image.size[1])
    for x in xrange(image.size[0]):
        for y in xrange(image.size[1]):
            difference = colorDifference((20, 20, 20), pixdata[x, y])
            pixdata[x, y] = (255, 255, 255) if difference < 200 else (0, 0, 0)
    image.save(constants.SHARED_TEST_PATH + filename, "JPEG")

    # Part 2, flood fill
    flood = Image.open(constants.SHARED_TEST_PATH + filename)
    flood = flood.convert("RGB")
    pixdata = flood.load()
    flood_result = largestContiguousRegion(flood)
    flood_result.save(constants.SHARED_TEST_PATH + filename, "PNG")
