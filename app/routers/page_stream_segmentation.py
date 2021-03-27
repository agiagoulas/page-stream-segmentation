from fastapi import APIRouter, UploadFile, File, status, Response
from loguru import logger
from pdf2image import convert_from_bytes
from pytesseract import image_to_string
from PIL import Image

from enum import Enum
import numpy as np
import fasttext
import cv2
import io
import traceback

from pss import model, model_img

router = APIRouter(prefix="/pss",
                   tags=["pss"],
                   responses={404: {
                       "description": "Not found"
                   }})

MODEL_IMAGE_PATH="./pss/models/image-only/Tobacco800_exp2_img_repeat-06.hdf5"
MODEL_IMAGE_PREV_PAGE_PATH="./pss/models/image-only/Tobacco800_exp2_prev-page_repeat-07.hdf5"
MODEL_TEXT_PATH="./pss/models/text-only/tobacco800_exp1_single-page_repeat-01.hdf5"
MODEL_TEXT_PREV_PAGE_PATH="./pss/models/text-only/tobacco800_exp1_prev-page_repeat_02-05.hdf5" 
FASTTEXT_WORD_VECTORS_PATH="./pss/models/fasttext/wiki.en.bin"            

logger.info("---Model Setup---")
logger.info("loading image models")
try:
    img_dim = (224,224)
    model_image = model_img.compile_model_singlepage(img_dim)
    model_image.load_weights(MODEL_IMAGE_PATH)
    model_image_prevpage = model_img.compile_model_prevpage(img_dim)
    model_image_prevpage.load_weights(MODEL_IMAGE_PREV_PAGE_PATH)
except Exception:
    logger.error("could not load image models")
    logger.error(traceback.format_exc())

logger.info("loading fasttext word vectors")
try:
    ft = fasttext.load_model(FASTTEXT_WORD_VECTORS_PATH)
    model.ft = ft
except Exception:
    logger.error("could not load fasttext word vectors")
    logger.error(traceback.format_exc())

logger.info("loading text models")
try:
    model_text = model.compile_model_singlepage()
    model_text.load_weights(MODEL_TEXT_PATH)
    model_text_prevpage = model.compile_model_prevpage()
    model_text_prevpage.load_weights(MODEL_TEXT_PREV_PAGE_PATH)
except Exception:
    logger.error("could not load text models")
    logger.error(traceback.format_exc())
logger.info("---Done---")



def convert_pdf_to_jpeg(pdf_bytes):
    return convert_from_bytes(pdf_bytes, fmt='jpeg')


def ocr_image_to_text(image_bytes_array, language='eng'):
    text = []

    for image in image_bytes_array:
        text.append(image_to_string(image, lang=language))

    return text


def generate_sequence(file_array, mode="text"):
    sequence = []

    prev_page_content = ""
    for count, current_page_content in enumerate(file_array):
        if mode == "text":
            sequence.append([current_page_content, prev_page_content, str(count)])
        else:
            sequence.append([str(count), "FirstPage", "", prev_page_content, ""])
        prev_page_content = current_page_content
    
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
                current_document.append(sequence[counter][2])
            elif prediction == 0:
                current_document.append(sequence[counter][2])
        else:
            first_page = False
            current_document = []
            current_document.append(sequence[counter][2])

        if counter == (len(y_predict) - 1):
            seperated_documents.append(current_document)

    return(seperated_documents)
        



class ModelType(str,Enum):
    single_page = "single_page"
    prev_page = "prev_page"


@router.post("/textModel/{model_type}/processDocument/")
async def upload_file(response: Response, model_type: ModelType, file: UploadFile = File(...)):

    logger.info("processing file: " + file.filename + " with " + model_type + " model" )

    # todo check for pdf file.content_type
    logger.debug("reading pdf file")
    try:
        file_bytes = await file.read()
    except Exception:
        logger.error(traceback.format_exc())
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"message": "could not read file"}

    logger.debug("converting pdf to jpeg images")
    try:
        file_jpegs_bytes = convert_pdf_to_jpeg(file_bytes)
    except Exception:
        logger.error(traceback.format_exc())
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"message": "could not convert pdf to images"}

    logger.debug("ocr processing images")
    try:
        file_texts = ocr_image_to_text(file_jpegs_bytes)
    except Exception:
        logger.error(traceback.format_exc())
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"message": "could not ocr process images"}

    logger.debug("generating sequence")
    try:
        sequence = generate_sequence(file_texts)
    except Exception:
        logger.error(traceback.format_exc())
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"message": "could not generate sequence"}

    if model_type == "single_page":
        logger.debug("generating predictions with single page text model")
        try:
            y_predict = model.predict(model=model_text, data=sequence, prev_page_generator=False, batch_size=256)
        except Exception:
            logger.error(traceback.format_exc())
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"message": "could not generate prediction"}
    elif model_type == "prev_page":
        logger.debug("generating predictions with prev page text model")
        try:
            y_predict = model.predict(model=model_text_prevpage, data=sequence, prev_page_generator=True, batch_size=256)
        except Exception:
            logger.error(traceback.format_exc())
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"message": "could not generate prediction"}




    print("processing predictions")
    processed_predictions = process_prediction(y_predict, sequence)
    
    print("predictions: ")
    print(processed_predictions)
    return(processed_predictions)




@router.post("/imageModel/processDocument")
async def upload_file(file: UploadFile = File(...)):

    file_bytes = await file.read()

    file_jpegs_bytes = convert_pdf_to_jpeg(file_bytes)

    file_images_resized_bytes = convert_image_to_resized(file_jpegs_bytes)

    sequence = generate_sequence(file_images_resized_bytes, mode="image")

    y_predict_image = np.round(model_image.predict_generator(model_img.ImageFeatureGenerator2(sequence, img_dim, image_binary_array = file_images_resized_bytes, prevpage = True)))

    print(y_predict_image)
    
    processed_predictions = process_prediction(y_predict_image, sequence)

    print(processed_predictions)
    return(processed_predictions)