from fastapi import APIRouter, UploadFile, File
from pdf2image import convert_from_bytes
from pytesseract import image_to_string
from PIL import Image

import numpy as np
import fasttext
import cv2
import io

from pss import model, model_img


router = APIRouter(prefix="/pss",
                   tags=["pss"],
                   responses={404: {
                       "description": "Not found"
                   }})



img_dim = (224,224)
model_prevpage_image = model_img.compile_model_prevpage(img_dim)
model_prevpage_image.load_weights("./pss/models/Tobacco800_exp2_prev-page_repeat-00.hdf5")



# print("-- Model Setup --")
# print("Loading fastText model")
# ft = fasttext.load_model("./pss/models/cc.en.300.bin")
# model.ft = ft
# print("Compiling model")
# model_prevpage = model.compile_model_prevpage()
# print("Loading model")
# model_prevpage.load_weights("./pss/models/tobacco800_exp1_prev-page_repeat-09.hdf5")
# print("-- Finished Setup --")

def convert_pdf_to_jpeg(pdf_bytes):
    return convert_from_bytes(pdf_bytes, fmt='jpeg')


def ocr_image_to_text(image_bytes_array, language='eng'):
    text = []

    for image in image_bytes_array:
        text.append(image_to_string(image, lang=language))

    return text


def generate_sequence(file_array, mode="text"):
    sequence = []

    prev_page = ""
    for count, file_content in enumerate(file_array):
        current_page = str(count)
        if mode == "text":
            sequence.append([current_page, "FirstPage", file_content, prev_page, ""])
        else:
            sequence.append([current_page, "FirstPage", "", prev_page, ""])
        prev_page = current_page
    
    return sequence


def otsu_tresholding_and_resizing(img):
    img_cv = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    gray, img_bin = cv2.threshold(img_cv,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    gray = cv2.bitwise_not(img_bin)
    # resized = cv2.resize(gray, img_dim, interpolation = cv2.INTER_AREA)
    resized = cv2.resize(gray, (225, 225), interpolation = cv2.INTER_AREA)
    img_pil = Image.fromarray(resized)
    
    return img_pil


def convert_image_to_resized(image_bytes_array):
    images = []

    for counter, image in enumerate(image_bytes_array):
        #image_object = otsu_tresholding_and_resizing(image)
        #images.append(image_object)
        images.append(image)

    return images


def process_prediction(y_predict, sequence):
    seperated_documents = []
    current_document = []

    first_page = True
    for counter, prediction in enumerate(y_predict):

        if not first_page:
            if prediction == 1:
                seperated_documents.append(current_document)
                current_document = []
                current_document.append(sequence[counter][0])
            elif prediction == 0:
                current_document.append(sequence[counter][0])
        else:
            first_page = False
            current_document = []
            current_document.append(sequence[counter][0])

        if counter == (len(y_predict) - 1):
            seperated_documents.append(current_document)

    return(seperated_documents)
        

@router.post("/textModel/processDocument")
async def upload_file(file: UploadFile = File(...)):

    file_bytes = await file.read()

    file_jpegs_bytes = convert_pdf_to_jpeg(file_bytes)

    file_texts = ocr_image_to_text(file_jpegs_bytes)

    sequence = generate_sequence(file_texts)

    y_predict = np.round(model_prevpage.predict_generator(model.TextFeatureGenerator2(sequence, batch_size=256)))

    processed_predictions = process_prediction(y_predict, sequence)
    
    return(processed_predictions)




@router.post("/imageModel/processDocument")
async def upload_file(file: UploadFile = File(...)):

    file_bytes = await file.read()

    file_jpegs_bytes = convert_pdf_to_jpeg(file_bytes)

    file_images_resized_bytes = convert_image_to_resized(file_jpegs_bytes)

    sequence = generate_sequence(file_images_resized_bytes, mode="image")

    y_predict_image = np.round(model_prevpage_image.predict_generator(model_img.ImageFeatureGenerator2(sequence, img_dim, image_binary_array = file_images_resized_bytes, prevpage = True)))

    print(y_predict_image)
    
    processed_predictions = process_prediction(y_predict_image, sequence)

    return(processed_predictions)