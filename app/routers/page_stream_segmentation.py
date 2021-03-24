from fastapi import APIRouter, UploadFile, File
from pdf2image import convert_from_bytes
from pytesseract import image_to_string
from PIL import Image

import numpy as np
import fasttext

from pss import model


router = APIRouter(prefix="/pss",
                   tags=["pss"],
                   responses={404: {
                       "description": "Not found"
                   }})


print("-- Model Setup --")
print("Loading fastText model")
ft = fasttext.load_model("./pss/models/cc.en.300.bin")
model.ft = ft
print("Compiling model")
model_prevpage = model.compile_model_prevpage()
print("Loading model")
model_prevpage.load_weights("./pss/models/tobacco800_exp1_prev-page_repeat-09.hdf5")
print("-- Finished Setup --")

def convert_pdf_to_jpeg(pdf_bytes):
    return convert_from_bytes(pdf_bytes, fmt='jpeg')


def ocr_image_to_text(image_bytes_array, language='eng'):
    text = []

    for image in image_bytes_array:
        text.append(image_to_string(image, lang=language))

    return text


def generate_sequence(file_texts):
    sequence = []

    for count, file_text in enumerate(file_texts):
        current_page = "page" + str(count)
        sequence.append([current_page, "FirstPage", file_text, "", ""])
    
    return sequence
        

@router.post("textModel/processDocument")
async def upload_file(file: UploadFile = File(...)):

    file_bytes = await file.read()

    file_jpegs_bytes = convert_pdf_to_jpeg(file_bytes)

    file_texts = ocr_image_to_text(file_jpegs_bytes)

    sequence = generate_sequence(file_texts)

    y_predict = np.round(model_prevpage.predict_generator(model.TextFeatureGenerator2(sequence, batch_size=256)))

    seperated_documents = []
    current_document = []

    first_page = True
    for counter, prediction in enumerate(y_predict):

        if not first_page:
            if prediction == 1:
                seperated_documents.append(current_document)
                current_document = []
                current_document.append(sequence[counter][0])
                print("Begin")
            elif prediction == 0:
                current_document.append(sequence[counter][0])
                print("Continue")
            print(sequence[counter][0])
        else:
            first_page = False
            current_document = []
            current_document.append(sequence[counter][0])
            print("Begin First")

        if counter == (len(y_predict) - 1):
            seperated_documents.append(current_document)

    return(seperated_documents)
    
