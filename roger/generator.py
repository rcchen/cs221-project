import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

import constants

def generateImage(text, font):
    image = Image.new("RGBA", (400, 400), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((10, 0), text, (0, 0, 0), font = font)
    draw = ImageDraw.Draw(image)
    image.save(constants.SHARED_TRAIN_PATH + text + ".png")

font = ImageFont.truetype(constants.SHARED_FONT_PATH + "collegeb.ttf", 370)
for i in xrange(1, 100):
    generateImage(str(i), font)    
