import os
from PIL import Image

image_dir = os.getcwd() + "/left_!/stroke_png/"
image_dir_resize = os.getcwd() + "/left_!/stroke_png_resize/"
for f in os.listdir(image_dir):
    filename = os.fsdecode(f)
    image = Image.open(image_dir + '/' + filename)
    image = image.resize((200, 200))
    image.save(image_dir_resize + filename,quality=200)