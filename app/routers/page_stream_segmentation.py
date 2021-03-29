from fastapi import APIRouter, UploadFile, File, status, Response
from loguru import logger
from pdf2image import convert_from_bytes
from pytesseract import image_to_string
from PIL import Image

from enum import Enum
import numpy as np
import fasttext
import cv2
import traceback

from app.pss import model, model_img
from app.data_models.prediction import Prediction, PredictionWrapper


router = APIRouter(prefix="/pss",
                   tags=["pss"],
                   responses={404: {
                       "description": "Not found"
                   }})


WORKING_DIR = "./app/pss/models/"
MODEL_TEXT = "tobacco800_text_single-page.hdf5"
MODEL_TEXT_PREV_PAGE = "tobacco800_text_prev-page.hdf5"
MODEL_IMAGE = "tobacco800_image_single-page.hdf5"
MODEL_IMAGE_PREV_PAGE = "tobacco800_image_prev-page.hdf5"
FASTTEXT_WORD_VECTORS = "wiki.en.bin"

POWER_TEXT_PREDICTION_PARAMETER = 0.4
POWER_IMAGE_PREDICTION_PARAMETER = 0.2

logger.info("---Model Setup---")
logger.info("loading image models")
try:
    img_dim = (224, 224)
    model_image = model_img.compile_model_singlepage(img_dim)
    model_image.load_weights(WORKING_DIR + MODEL_IMAGE)
    model_image_prevpage = model_img.compile_model_prevpage(img_dim)
    model_image_prevpage.load_weights(WORKING_DIR + MODEL_IMAGE_PREV_PAGE)
except Exception:
    logger.error("could not load image models")
    logger.error(traceback.format_exc())

logger.info("loading fasttext word vectors")
try:
    ft = fasttext.load_model(WORKING_DIR + FASTTEXT_WORD_VECTORS)
    model.ft = ft
except Exception:
    logger.error("could not load fasttext word vectors")
    logger.error(traceback.format_exc())

logger.info("loading text models")
try:
    model_text = model.compile_model_singlepage()
    model_text.load_weights(WORKING_DIR + MODEL_TEXT)
    model_text_prevpage = model.compile_model_prevpage()
    model_text_prevpage.load_weights(WORKING_DIR + MODEL_TEXT_PREV_PAGE)
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


def generate_sequence(file_array):
    sequence = []
    prev_page_content = ""
    for count, current_page_content in enumerate(file_array):
        sequence.append([str(count), current_page_content, prev_page_content])
        prev_page_content = current_page_content
    return sequence


# TODO add tresholding
def otsu_tresholding_and_resizing(img):
    img_cv = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    gray, img_bin = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    gray = cv2.bitwise_not(img_bin)
    # resized = cv2.resize(gray, img_dim, interpolation = cv2.INTER_AREA)
    resized = cv2.resize(gray, (225, 225), interpolation=cv2.INTER_AREA)
    img_pil = Image.fromarray(resized)
    return img_pil


def convert_image_to_resized(image_bytes_array):
    images = []

    for counter, image in enumerate(image_bytes_array):
        # image_object = otsu_tresholding_and_resizing(image)
        # images.append(image_object)
        images.append(image)

    return images


def process_prediction_to_corresponding_pages(y_predict, sequence):
    seperated_documents = []
    current_document = []

    first_page = True
    for counter, prediction in enumerate(y_predict):

        if not first_page:
            if prediction == 1:
                seperated_documents.append(current_document)
                current_document = []
                current_document.append("page " + sequence[counter][0])
            elif prediction == 0:
                current_document.append("page " + sequence[counter][0])
        else:
            first_page = False
            current_document = []
            current_document.append("page " + sequence[counter][0])

        if counter == (len(y_predict) - 1):
            seperated_documents.append(current_document)

    return (seperated_documents)


def convert_numpy_to_int_list(numpy_array):
    int_list = numpy_array.astype(int).tolist()
    prediction_list = []
    for value in int_list:
        prediction_list.append(value[0])
    return prediction_list


def convert_numpy_to_float_list(numpy_array):
    int_list = numpy_array.astype(float).tolist()
    prediction_list = []
    for value in int_list:
        prediction_list.append(value[0])
    return prediction_list


class ModelType(str, Enum):
    single_page = "single_page"
    prev_page = "prev_page"


@router.post("/textModel/{model_type}/processDocument/", response_model=PredictionWrapper)
async def process_document_with_text_model(response: Response, model_type: ModelType, file: UploadFile = File(...)):
    logger.info("processing file: " + file.filename + " with " + model_type + " model")

    if not file.content_type == "application/pdf":
        logger.warning("submitted file is no pdf")
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"message": "submitted file is no pdf"}

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
        used_model_name = MODEL_TEXT
        try:
            y = model.predict_without_rounding(model=model_text, data=sequence, prev_page_generator=False)
            y_exact = convert_numpy_to_float_list(y)
            y_predict_numpy = np.round(y)
            y_predict = convert_numpy_to_int_list(y_predict_numpy)
            corresponding_pages = process_prediction_to_corresponding_pages(y_predict_numpy, sequence)
            prediction = Prediction(model=used_model_name, y_predict=y_predict, y_exact=y_exact, corresponding_pages=corresponding_pages)
        except Exception:
            logger.error(traceback.format_exc())
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"message": "could not generate prediction"}
    elif model_type == "prev_page":
        logger.debug("generating predictions with prev page text model")
        used_model_name = MODEL_TEXT_PREV_PAGE
        try:
            y = model.predict_without_rounding(model=model_text_prevpage, data=sequence, prev_page_generator=True)
            y_exact = convert_numpy_to_float_list(y)
            y_predict_numpy = np.round(y)
            y_predict = convert_numpy_to_int_list(y_predict_numpy)
            corresponding_pages = process_prediction_to_corresponding_pages(y_predict_numpy, sequence)
            prediction = Prediction(model=used_model_name, y_predict=y_predict, y_exact=y_exact, corresponding_pages=corresponding_pages)
        except Exception:
            logger.error(traceback.format_exc())
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"message": "could not generate prediction"}

    predictions = PredictionWrapper(file_name=file.filename, predictions=[prediction])
    logger.info(predictions)
    return predictions


@router.post("/imageModel/{model_type}/processDocument/", response_model=PredictionWrapper)
async def process_document_with_image_model(response: Response, model_type: ModelType, file: UploadFile = File(...)):
    logger.info("processing file: " + file.filename + " with " + model_type + " model")

    if not file.content_type == "application/pdf":
        logger.warning("submitted file is no pdf")
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"message": "submitted file is no pdf"}

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

    logger.debug("resizing image bytes")
    try:
        file_images_resized_bytes = convert_image_to_resized(file_jpegs_bytes)
    except Exception:
        logger.error(traceback.format_exc())
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"message": "could not resize image bytes"}

    logger.debug("generating sequence")
    try:
        sequence = generate_sequence(file_images_resized_bytes)
    except Exception:
        logger.error(traceback.format_exc())
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"message": "could not generate sequence"}

    if model_type == "single_page":
        logger.debug("generating predictions with single page image model")
        used_model_name = MODEL_IMAGE
        try:
            y = model_img.predict_without_rounding(model=model_image, data=sequence, img_dim=img_dim, prev_page_generator=False)
            y_exact = convert_numpy_to_float_list(y)
            y_predict_numpy = np.round(y)
            y_predict = convert_numpy_to_int_list(y_predict_numpy)
            corresponding_pages = process_prediction_to_corresponding_pages(y_predict_numpy, sequence)
            prediction = Prediction(model=used_model_name, y_predict=y_predict, y_exact=y_exact, corresponding_pages=corresponding_pages)
        except Exception:
            logger.error(traceback.format_exc())
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"message": "could not generate prediction"}
    elif model_type == "prev_page":
        logger.debug("generating predictions with prev page image model")
        used_model_name = MODEL_IMAGE_PREV_PAGE
        try:
            y = model_img.predict_without_rounding(model=model_image_prevpage, data=sequence, img_dim=img_dim, prev_page_generator=True)
            y_exact = convert_numpy_to_float_list(y)
            y_predict_numpy = np.round(y)
            y_predict = convert_numpy_to_int_list(y_predict_numpy)
            corresponding_pages = process_prediction_to_corresponding_pages(y_predict_numpy, sequence)
            prediction = Prediction(model=used_model_name, y_predict=y_predict, y_exact=y_exact, corresponding_pages=corresponding_pages)
        except Exception:
            logger.error(traceback.format_exc())
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"message": "could not generate prediction"}

    predictions = PredictionWrapper(file_name=file.filename, predictions=[prediction])
    logger.info(predictions)
    return predictions


@router.post("/combinedModels/{text_model_type}/{image_model_type}/processDocument/", response_model=PredictionWrapper)
async def process_document_with_text_model(response: Response, text_model_type: ModelType = "prev_page", image_model_type: ModelType = "single_page", file: UploadFile = File(...)):
    logger.info("processing file: " + file.filename + " with text model: " + text_model_type + " and image model: " + image_model_type)

    if not file.content_type == "application/pdf":
        logger.warning("submitted file is no pdf")
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"message": "submitted file is no pdf"}

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

    logger.debug("resizing image bytes")
    try:
        file_images_resized_bytes = convert_image_to_resized(file_jpegs_bytes)
    except Exception:
        logger.error(traceback.format_exc())
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"message": "could not resize image bytes"}

    logger.debug("ocr processing images")
    try:
        file_texts = ocr_image_to_text(file_jpegs_bytes)
    except Exception:
        logger.error(traceback.format_exc())
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"message": "could not ocr process images"}

    logger.debug("generating sequences")
    try:
        text_sequence = generate_sequence(file_texts)
        image_sequence = generate_sequence(file_images_resized_bytes)
    except Exception:
        logger.error(traceback.format_exc())
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"message": "could not generate sequence"}

    # text model prediction
    if text_model_type == "single_page":
        logger.debug("generating predictions with single page text model")
        used_model_name = MODEL_TEXT
        try:
            text_y = model.predict_without_rounding(model=model_text, data=text_sequence, prev_page_generator=False)
            text_y_exact = convert_numpy_to_float_list(text_y)
            text_y_predict_numpy = np.round(text_y)
            text_y_predict = convert_numpy_to_int_list(text_y_predict_numpy)
            text_corresponding_pages = process_prediction_to_corresponding_pages(text_y_predict_numpy, text_sequence)
            text_prediction = Prediction(model=used_model_name, y_predict=text_y_predict, y_exact=text_y_exact, corresponding_pages=text_corresponding_pages)
        except Exception:
            logger.error(traceback.format_exc())
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"message": "could not generate prediction"}
    elif text_model_type == "prev_page":
        logger.debug("generating predictions with prev page text model")
        used_model_name = MODEL_TEXT_PREV_PAGE
        try:
            text_y = model.predict_without_rounding(model=model_text_prevpage, data=text_sequence, prev_page_generator=True)
            text_y_exact = convert_numpy_to_float_list(text_y)
            text_y_predict_numpy = np.round(text_y)
            text_y_predict = convert_numpy_to_int_list(text_y_predict_numpy)
            text_corresponding_pages = process_prediction_to_corresponding_pages(text_y_predict_numpy, text_sequence)
            text_prediction = Prediction(model=used_model_name, y_predict=text_y_predict, y_exact=text_y_exact, corresponding_pages=text_corresponding_pages)
        except Exception:
            logger.error(traceback.format_exc())
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"message": "could not generate prediction"}

    # image model prediction
    if image_model_type == "single_page":
        logger.debug("generating predictions with single page image model")
        used_model_name = MODEL_IMAGE
        try:
            image_y = model_img.predict_without_rounding(model=model_image, data=image_sequence, img_dim=img_dim, prev_page_generator=False)
            image_y_exact = convert_numpy_to_float_list(image_y)
            image_y_predict_numpy = np.round(image_y)
            image_y_predict = convert_numpy_to_int_list(image_y_predict_numpy)
            image_corresponding_pages = process_prediction_to_corresponding_pages(image_y_predict_numpy, image_sequence)
            image_prediction = Prediction(model=used_model_name, y_predict=image_y_predict, y_exact=image_y_exact, corresponding_pages=image_corresponding_pages)
        except Exception:
            logger.error(traceback.format_exc())
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"message": "could not generate prediction"}
    elif image_model_type == "prev_page":
        logger.debug("generating predictions with prev page image model")
        used_model_name = MODEL_IMAGE_PREV_PAGE
        try:
            image_y = model_img.predict_without_rounding(model=model_image_prevpage, data=image_sequence, img_dim=img_dim, prev_page_generator=True)
            image_y_exact = convert_numpy_to_float_list(image_y)
            image_y_predict_numpy = np.round(image_y)
            image_y_predict = convert_numpy_to_int_list(image_y_predict_numpy)
            image_corresponding_pages = process_prediction_to_corresponding_pages(image_y_predict_numpy, image_sequence)
            image_prediction = Prediction(model=used_model_name, y_predict=image_y_predict, y_exact=image_y_exact, corresponding_pages=image_corresponding_pages)
        except Exception:
            logger.error(traceback.format_exc())
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"message": "could not generate prediction"}

    logger.debug("generating combined predictions from both models")
    try:
        text_probability = np.concatenate([1 - text_y_predict_numpy, text_y_predict_numpy], axis=1)
        image_probability = np.concatenate([1 - image_y_predict_numpy, image_y_predict_numpy], axis=1)
        combined_y_predict_numpy = np.argmax(np.power(text_probability, POWER_TEXT_PREDICTION_PARAMETER) * np.power(image_probability, POWER_IMAGE_PREDICTION_PARAMETER), axis=1)
        combined_corresponding_pages = process_prediction_to_corresponding_pages(combined_y_predict_numpy, text_sequence)
        int_list = combined_y_predict_numpy.astype(int).tolist()
        combined_y_predict = []
        for value in int_list:
            combined_y_predict.append(value)
        combined_prediction = Prediction(model="combined model", y_predict=combined_y_predict, corresponding_pages=combined_corresponding_pages)
    except Exception:
        logger.error(traceback.format_exc())
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"message": "could not generate combined prediction"}

    predictions = PredictionWrapper(file_name=file.filename, predictions=[combined_prediction, text_prediction, image_prediction])
    logger.info(predictions)
    return predictions
