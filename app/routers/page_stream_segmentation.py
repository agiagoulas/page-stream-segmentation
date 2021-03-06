from fastapi import APIRouter, UploadFile, File, status, Response
from loguru import logger
from pdf2image import convert_from_bytes
from pytesseract import image_to_string
from PIL import Image
from io import BytesIO    
from enum import Enum
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizerFast

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

# enables or disables model functionality
ENABLE_GRU_TEXT_MODELS = True 
ENABLE_VGG16_IMAGE_MODELS = True
ENABLE_BERT_TEXT_MODELS = True

WORKING_DIR = "./app/pss/models/"
MODEL_TEXT = "tobacco800_text_single-page.hdf5"
MODEL_TEXT_PREV_PAGE = "tobacco800_text_prev-page.hdf5"
MODEL_IMAGE = "tobacco800_image_single-page.hdf5"
MODEL_IMAGE_PREV_PAGE = "tobacco800_image_prev-page.hdf5"
FASTTEXT_WORD_VECTORS = "wiki.en.bin"
BERT_MODEL_TEXT = "agiagoulas/bert-pss"


if ENABLE_VGG16_IMAGE_MODELS:
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
    logger.info("finished loading image models")


if ENABLE_GRU_TEXT_MODELS:
    logger.info("loading fasttext word vectors")
    try:
        ft = fasttext.load_model(WORKING_DIR + FASTTEXT_WORD_VECTORS)
        model.ft = ft
    except Exception:
        logger.error("could not load fasttext word vectors")
        logger.error(traceback.format_exc())
    logger.info("finished loading fasttext word vectors")

    logger.info("loading text models")
    try:
        model_text = model.compile_model_singlepage()
        model_text.load_weights(WORKING_DIR + MODEL_TEXT)
        model_text_prevpage = model.compile_model_prevpage()
        model_text_prevpage.load_weights(WORKING_DIR + MODEL_TEXT_PREV_PAGE)
    except Exception:
        logger.error("could not load text models")
        logger.error(traceback.format_exc())
    logger.info("finished loading text models")

if ENABLE_BERT_TEXT_MODELS:
    logger.info("loading bert text models")
    bert_model_text = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_TEXT)
    bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    logger.info("finished loading bert text models")


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


def process_prediction_to_corresponding_pages_list(y_predict):
    seperated_documents = []
    current_document = []

    first_page = True
    for counter, prediction in enumerate(y_predict):

        if not first_page:
            if prediction == 1:
                seperated_documents.append(current_document)
                current_document = []
                current_document.append("page " + str(counter))
            elif prediction == 0:
                current_document.append("page " + str(counter))
        else:
            first_page = False
            current_document = []
            current_document.append("page " + str(counter))

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


if ENABLE_BERT_TEXT_MODELS:
    @router.post("/bertTextModel/processDocument/", response_model=PredictionWrapper)
    async def process_document_with_text_model(response: Response, file: UploadFile = File(...)):
        logger.info("processing file: " + file.filename + " with bert text model")

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

        logger.debug("generating bert model predictions")
        try:
            used_model_name = BERT_MODEL_TEXT
            y_predict=[]
            for page in file_texts:
                inputs = bert_tokenizer(page, padding=True, truncation=True, return_tensors="pt")
                outputs = bert_model_text(**inputs)
                y_predict.append(outputs.logits.argmax(-1).item())
            corresponding_pages = process_prediction_to_corresponding_pages_list(y_predict)
            prediction = Prediction(model=used_model_name, y_predict=y_predict, corresponding_pages=corresponding_pages)
        except Exception:
            logger.error(traceback.format_exc())
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"message": "could not generate bert model predictions"}

        predictions = PredictionWrapper(file_name=file.filename, predictions=[prediction])
        logger.info(predictions)
        return predictions
  

if ENABLE_GRU_TEXT_MODELS:
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
            used_model = model_text
            prev_page_generator = False
        elif model_type == "prev_page":
            logger.debug("generating predictions with prev page text model")
            used_model_name = MODEL_TEXT_PREV_PAGE
            used_model = model_text_prevpage
            prev_page_generator = True

        try:
            y = model.predict_without_rounding(model=used_model, data=sequence, prev_page_generator=prev_page_generator)
            y_exact = convert_numpy_to_float_list(y)
            y_predict_numpy = np.round(y)
            y_predict = convert_numpy_to_int_list(y_predict_numpy)
            corresponding_pages = process_prediction_to_corresponding_pages_list(y_predict)
            prediction = Prediction(model=used_model_name, y_predict=y_predict, y_exact=y_exact, corresponding_pages=corresponding_pages)
        except Exception:
            logger.error(traceback.format_exc())
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"message": "could not generate prediction"}

        predictions = PredictionWrapper(file_name=file.filename, predictions=[prediction])
        logger.info(predictions)
        return predictions


if ENABLE_VGG16_IMAGE_MODELS:
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

        logger.debug("generating sequence")
        try:
            sequence = generate_sequence(file_jpegs_bytes)
        except Exception:
            logger.error(traceback.format_exc())
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"message": "could not generate sequence"}

        if model_type == "single_page":
            logger.debug("generating predictions with single page image model")
            used_model_name = MODEL_IMAGE
            used_model = model_image
            prev_page_generator = False
        elif model_type == "prev_page":
            logger.debug("generating predictions with prev page image model")
            used_model_name = MODEL_IMAGE_PREV_PAGE
            used_model = model_image_prevpage
            prev_page_generator = True

        try:
            y = model_img.predict_without_rounding(model=used_model, data=sequence, img_dim=img_dim, prev_page_generator=prev_page_generator)
            y_exact = convert_numpy_to_float_list(y)
            y_predict_numpy = np.round(y)
            y_predict = convert_numpy_to_int_list(y_predict_numpy)
            corresponding_pages = process_prediction_to_corresponding_pages_list(y_predict)
            prediction = Prediction(model=used_model_name, y_predict=y_predict, y_exact=y_exact, corresponding_pages=corresponding_pages)
        except Exception:
            logger.error(traceback.format_exc())
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"message": "could not generate prediction"}

        predictions = PredictionWrapper(file_name=file.filename, predictions=[prediction])
        logger.info(predictions)
        return predictions


if ENABLE_VGG16_IMAGE_MODELS and ENABLE_GRU_TEXT_MODELS:
    @router.post("/combinedModels/{text_model_type}/{image_model_type}/processDocument/", response_model=PredictionWrapper)
    async def process_document_with_combined_models(response: Response, text_model_type: ModelType = "prev_page", image_model_type: ModelType = "single_page", file: UploadFile = File(...)):
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
            image_sequence = generate_sequence(file_jpegs_bytes)
        except Exception:
            logger.error(traceback.format_exc())
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"message": "could not generate sequence"}

        # text model prediction
        if text_model_type == "single_page":
            logger.debug("generating predictions with single page text model")
            used_text_model_name = MODEL_TEXT
            used_text_model = model_text
            text_prev_page_generator = False   
        elif text_model_type == "prev_page":
            logger.debug("generating predictions with prev page text model")
            used_text_model_name = MODEL_TEXT_PREV_PAGE
            used_text_model = model_text_prevpage
            text_prev_page_generator = True

        try:
            text_y = model.predict_without_rounding(model=used_text_model, data=text_sequence, prev_page_generator=text_prev_page_generator)
            text_y_exact = convert_numpy_to_float_list(text_y)
            text_y_predict_numpy = np.round(text_y)
            text_y_predict = convert_numpy_to_int_list(text_y_predict_numpy)
            text_corresponding_pages = process_prediction_to_corresponding_pages_list(text_y_predict)
            text_prediction = Prediction(model=used_text_model_name, y_predict=text_y_predict, y_exact=text_y_exact, corresponding_pages=text_corresponding_pages)
        except Exception:
            logger.error(traceback.format_exc())
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"message": "could not generate prediction"}

        # image model prediction
        if image_model_type == "single_page":
            logger.debug("generating predictions with single page image model")
            used_image_model_name = MODEL_IMAGE
            used_image_model = model_image
            image_prev_page_generator = False
        elif image_model_type == "prev_page":
            logger.debug("generating predictions with prev page image model")
            used_image_model_name = MODEL_IMAGE_PREV_PAGE
            used_image_model = model_image_prevpage
            image_prev_page_generator = True

        try:
            image_y = model_img.predict_without_rounding(model=used_image_model, data=image_sequence, img_dim=img_dim, prev_page_generator=image_prev_page_generator)
            image_y_exact = convert_numpy_to_float_list(image_y)
            image_y_predict_numpy = np.round(image_y)
            image_y_predict = convert_numpy_to_int_list(image_y_predict_numpy)
            image_corresponding_pages = process_prediction_to_corresponding_pages_list(image_y_predict)
            image_prediction = Prediction(model=used_image_model_name, y_predict=image_y_predict, y_exact=image_y_exact, corresponding_pages=image_corresponding_pages)
        except Exception:
            logger.error(traceback.format_exc())
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"message": "could not generate prediction"}

        logger.debug("generating combined predictions from both models")
        # selection of correct power parameters
        if text_model_type == "single_page" and image_model_type == "single_page":
            text_prediction_power_parameter = 0.1
            image_prediction_power_parameter = 0.3
        elif text_model_type == "single_page" and image_model_type == "prev_page":
            text_prediction_power_parameter = 0.2
            image_prediction_power_parameter = 0.3
        elif text_model_type == "prev_page" and image_model_type == "single_page":
            text_prediction_power_parameter = 0.1
            image_prediction_power_parameter = 0.7
        elif text_model_type == "prev_page" and image_model_type == "prev_page":
            text_prediction_power_parameter = 0.1
            image_prediction_power_parameter = 0.6

        try:
            text_probability = np.concatenate([1 - text_y_predict_numpy, text_y_predict_numpy], axis=1)
            image_probability = np.concatenate([1 - image_y_predict_numpy, image_y_predict_numpy], axis=1)
            combined_y_predict_numpy = np.argmax(np.power(text_probability, text_prediction_power_parameter) * np.power(image_probability, image_prediction_power_parameter), axis=1)
            combined_corresponding_pages = process_prediction_to_corresponding_pages_list(combined_y_predict_numpy)
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


if ENABLE_BERT_TEXT_MODELS and ENABLE_VGG16_IMAGE_MODELS:
    @router.post("/combinedModelsBert/{image_model_type}/processDocument/", response_model=PredictionWrapper)
    async def process_document_with_combined_bert_models(response: Response, image_model_type: ModelType = "single_page", file: UploadFile = File(...)):
        logger.info("processing file: " + file.filename + " with bert text model and image model: " + image_model_type)

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
            image_sequence = generate_sequence(file_jpegs_bytes)
        except Exception:
            logger.error(traceback.format_exc())
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"message": "could not generate sequence"}

        if image_model_type == "single_page":
            logger.debug("generating predictions with single page image model")
            used_image_model_name = MODEL_IMAGE
            used_image_model = model_image
            image_prev_page_generator = False
        elif image_model_type == "prev_page":
            logger.debug("generating predictions with prev page image model")
            used_image_model_name = MODEL_IMAGE_PREV_PAGE
            used_image_model = model_image_prevpage
            image_prev_page_generator = True
        try:
            image_y = model_img.predict_without_rounding(model=used_image_model, data=image_sequence, img_dim=img_dim, prev_page_generator=image_prev_page_generator)
            image_y_exact = convert_numpy_to_float_list(image_y)
            image_y_predict_numpy = np.round(image_y)
            image_y_predict = convert_numpy_to_int_list(image_y_predict_numpy)
            image_corresponding_pages = process_prediction_to_corresponding_pages_list(image_y_predict)
            image_prediction = Prediction(model=used_image_model_name, y_predict=image_y_predict, y_exact=image_y_exact, corresponding_pages=image_corresponding_pages)
        except Exception:
            logger.error(traceback.format_exc())
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"message": "could not generate prediction"}

        logger.debug("generating bert model predictions")
        try:
            used_model_name = BERT_MODEL_TEXT
            text_y_predict=[]
            for page in file_texts:
                inputs = bert_tokenizer(page, padding=True, truncation=True, return_tensors="pt")
                outputs = bert_model_text(**inputs)
                text_y_predict.append(outputs.logits.argmax(-1).item())
            text_corresponding_pages = process_prediction_to_corresponding_pages_list(text_y_predict)
            text_prediction = Prediction(model=used_model_name, y_predict=text_y_predict, corresponding_pages=text_corresponding_pages)
        except Exception:
            logger.error(traceback.format_exc())
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"message": "could not generate bert model predictions"}

        logger.debug("generating combined predictions from both models")
        text_prediction_power_parameter = 0.1
        image_prediction_power_parameter = 0.1
        try:
            prediction_bert = np.array(text_y_predict).reshape(-1, 1)
            text_probability = np.concatenate([1 - prediction_bert, prediction_bert], axis=1)
            image_probability = np.concatenate([1 - image_y_predict_numpy, image_y_predict_numpy], axis=1)
            combined_y_predict_numpy = np.argmax(np.power(text_probability, text_prediction_power_parameter) * np.power(image_probability, image_prediction_power_parameter), axis=1)
            combined_corresponding_pages = process_prediction_to_corresponding_pages_list(combined_y_predict_numpy)
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
