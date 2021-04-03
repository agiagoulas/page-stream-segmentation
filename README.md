# Multi Modal Page Stream Segmentation 

Implementation of Multi-Modal Page Stream Segmentation with CNN Networks as a Python Service

**Additional Resources**
- **model_training** contains jupyter notebooks to train the CNN models
- **document_stream_builder** contains a script to build document streams from pdf files

## Requirements

### Needed Installations

- [Poppler](https://poppler.freedesktop.org/) (PDF Rendering Library)   
    ```brew install poppler```
- [CMake](https://cmake.org/)
    ```brew install cmake```
- [tesseract](https://github.com/tesseract-ocr/tesseract)
    ```brew install tesseract```
- python
- [poetry](https://python-poetry.org/) (Dependency Management)

### Fasttext Word Vectors

Download english fasttext [wiki word vectors](https://fasttext.cc/docs/en/pretrained-vectors.html) under this [link](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip) and put **wiki.en.bin** in *./app/pss/models/*.

### Install Python Dependencies
```
poetry install
```

### Start Uvicorn Server
```
uvicorn app.main:app --reload
```

## Routes

OpenAPI Documentation http://localhost:8000/openapi.json

SwaggerUI Instance http://localhost:8000/docs

>For all routes the model_type can always be *single_page* or *prev_page*. 
>
>*single_page* selects the model, that was trained only with the current page as input data. 
>
>*prev_page* selects the model, that was trained with a pair of consecutive pages as input data.

The PDF Documents for all routes need to be sent at the request body as **form-data** with the key being **file**

### Text Only Processing

**POST** <localhost:8000/pss/textModel/{model_type}/processDocument/> 

Process PDF Documents with models that only consider the text data.

**Model Performance**

Model | Accuracy | Kappa
--- | --- | ---
single_page | 0,826255 | 0,627790
*prev_page* | *0,830116* | *0,641725*

### Text Only Processing with Transformers

**POST** <localhost:8000/pss/bertTextModel/processDocument/> 

Process PDF Documents with a transformer bert model, that only considers the text data.

Model | Accuracy | Kappa
--- | --- | --- 
*bert-based-uncased* | *0,915058* | *0,824828*

### Image Only Processing

**POST** <localhost:8000/pss/imageModel/{model_type}/processDocument/>

Process PDF Documents with models that only consider the image data. 
 
**Model Performance**

Model | Accuracy | Kappa
--- | --- | ---
single_page | 0,926641 | 0,847236
*prev_page* | *0,934363* | *0,863316*

### Combined Multi-Modal Processing

**POST** <localhost:8000/pss/combinedModels/{text_model_type}/{image_model_type}/processDocument/>

Process PDF Documents with text and image models and combine the output for a multi-modal PSS prediction.

Text Model | Image Model | Accuracy | Kappa
--- | --- | --- | ---
single_page | single_page | 0,926641 | 0,847236
*single_page* | *prev_page* | *0,942085* | *0,879059*
prev_page | single_page | 0,918919 | 0,830682
prev_page | prev_page | 0,938224 | 0,871176

### Combined Multi-Modal Processing with Transformers (Text)

**POST** <localhost:8000/pss/combinedModels/bert_model/{image_model_type}/processDocument/>

Process PDF Documents with bert text and image models and combine the output for a multi-modal PSS prediction.

The bert based model is hosted at the [huggingface model repository](https://huggingface.co/agiagoulas/bert-pss)

Text Model | Image Model | Accuracy | Kappa
--- | --- | --- | ---
bert-based-uncased | single_page | 0,926641 | 0,850575
*bert-based-uncased* | *prev_page* | *0,934363* | *0,866669* 

## Notice

This repository builds onto the works of Wiedemann & Heyer 2019:
>Wiedemann, G., Heyer, G. Multi-modal page stream segmentation with convolutional neural networks.
>Lang Resources & Evaluation (2019). https://doi.org/10.1007/s10579-019-09476-2

The Model Training was performed with the Tobacco800 Dataset:
(Model Performance was measured with a test subset)
>David Doermann, Tobacco 800 Dataset (Tobacco800) http://tc11.cvc.uab.es/datasets/Tobacco800_1



