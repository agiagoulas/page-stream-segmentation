#!/usr/bin/env python

from os import listdir, path
from os.path import isfile, join
import random
import traceback
import argparse
from PyPDF2 import PdfFileMerger


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="input directory", default="./input/")
parser.add_argument("-o", "--output", help="output directory", default="./output/")
parser.add_argument("-r", "--random", help="disable random document order", default="True")
parser.add_argument("-l", "--limit", help="limit amount of processed documents")
args = parser.parse_args()


input_documents_list = [f for f in listdir(args.input) if isfile(join(args.input, f)) and f.endswith('.pdf')]
if args.random:
  random.shuffle(input_documents_list)

output_files_list = [f for f in listdir(args.output) if isfile(join(args.output, f))]

i = 0
while path.exists(args.output + "documentStream-%s.pdf" % i):
    i += 1

document_count = 0
complete_page_count = 0
merger = PdfFileMerger()

ground_truth_pages = []
    

for counter, pdf in enumerate(input_documents_list):
    if args.limit:
      if int(args.limit) == counter:
        break

    try:
      merger.append(open(args.input + pdf, 'rb'))
      page_count = len(merger.pages) - complete_page_count
      document_pages = []
      for q in range(page_count): 
        document_pages.append("page" + str(complete_page_count + q))
      ground_truth_pages.append(document_pages)
      complete_page_count = len(merger.pages)
      document_count += 1
    except Exception as e:
      print(traceback.format_exc())
      print("Could not append document: " + args.input + pdf)


if document_count == len(ground_truth_pages):
  with open(args.output + "documentStream-%s.pdf" % i, "wb") as f:
    merger.write(f)

  with open(args.output + "groundTruthPages-%s.txt" % i, "w") as g:
    g.write(str(ground_truth_pages))

else:
  print("Error in document processing!")
