import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

import constants

NUM_DIGITS = 10
ANGLE_RANGE = 45

def generateImage(text, font):
    image = Image.new("RGBA", (400, 400), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((100, 0), text, (0, 0, 0), font = font)
    draw = ImageDraw.Draw(image)
    image.save(constants.SHARED_TRAIN_PATH + text + ".png")

def generateRotations(base):
    image = Image.open(constants.SHARED_TRAIN_PATH + base + ".png")
    for i in xrange(-ANGLE_RANGE, ANGLE_RANGE):
        result = image.rotate(i)
        result.save(constants.SHARED_TRAIN_PATH + base + "_" + str(i) + ".png")

font = ImageFont.truetype(constants.SHARED_FONT_PATH + "collegeb.ttf", 400)
for i in xrange(NUM_DIGITS):
    digit = str(i)
    generateImage(digit, font)
    generateRotations(digit)
