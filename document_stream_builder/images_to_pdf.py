from os import listdir, path
from os.path import isfile, join
import random
import traceback
import argparse

from PIL import Image
from fpdf import FPDF

input_dir = "./image_input/"


input_documents_list = [f for f in listdir(input_dir) if isfile(join(input_dir, f)) and f.endswith('.png')]

input_documents_list.sort()







cover = Image.open(input_dir + str(input_documents_list[0]))
width, height = cover.size

pdf = FPDF(unit = "pt", format = [width, height])

for page in input_documents_list:
    print(page)
    pdf.add_page()
    pdf.image(input_dir + str(page), 0, 0)

pdf.output(input_dir + "tobacco800.pdf", "F")