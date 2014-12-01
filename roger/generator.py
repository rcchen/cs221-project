import numpy, PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

import constants

NUM_DIGITS = 10
ANGLE_RANGE = 45
IMAGE_SIZE = 100

# Taken from http://stackoverflow.com/questions/14177744/how-does-perspective-transformation-work-in-pil
def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = numpy.matrix(matrix, dtype=numpy.float)
    B = numpy.array(pb).reshape(8)

    res = numpy.dot(numpy.linalg.inv(A.T * A) * A.T, B)
    return numpy.array(res).reshape(8)

def generateImage(text, font):
    image = Image.new("RGBA", (IMAGE_SIZE, IMAGE_SIZE), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((100, 0), text, (0, 0, 0), font = font)
    draw = ImageDraw.Draw(image)
    image.save(constants.SHARED_TRAIN_PATH + text + ".png")

def generateRotations(base):
    image = Image.open(constants.SHARED_TRAIN_PATH + base + ".png")
    for i in xrange(-ANGLE_RANGE, ANGLE_RANGE):
        result = image.rotate(i)
        fff = Image.new('RGBA', image.size, (255, 255, 255, 255))
        out = Image.composite(result, fff, result)
        out.save(constants.SHARED_TRAIN_PATH + base + "_" + str(i) + ".png")

def generateDistortions(base):
    image = Image.open(constants.SHARED_TRAIN_PATH + base + ".png")
    width, height = image.size
    for i in xrange(20):
        j = i * -5
        coefficients = find_coeffs(
            [(50, 50 + j), (350, 80), (350, 220), (IMAGE_SIZE * 0.125, IMAGE_SIZE * 0.75 - j)],
            [(0, 0), (IMAGE_SIZE, 0), (IMAGE_SIZE, IMAGE_SIZE), (0, IMAGE_SIZE)])
        result = image.transform((width, height), Image.PERSPECTIVE, coefficients, Image.BICUBIC)
        result.save(constants.SHARED_TRAIN_PATH + base + "_skew_" + str(i) + ".png")

font = ImageFont.truetype(constants.SHARED_FONT_PATH + "collegeb.ttf", IMAGE_SIZE)
for i in xrange(NUM_DIGITS):
    digit = str(i)
    generateImage(digit, font)
    generateRotations(digit)
    # generateDistortions(digit)
