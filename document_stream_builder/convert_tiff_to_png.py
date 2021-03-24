import os
from os import listdir, path
from os.path import isfile, join
from PIL import Image

input_dir = "./image_input/"

input_documents_list = [f for f in listdir(input_dir) if isfile(join(input_dir, f)) and f.endswith('.tif')]

for name in input_documents_list:
    outfile = input_dir + name + ".png"
    im = Image.open(input_dir + name)
    im.thumbnail(im.size)
    im.save(outfile, "PNG", quality=100)
