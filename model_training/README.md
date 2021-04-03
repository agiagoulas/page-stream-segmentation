# Model Training Jupyter Notebooks

## Prework

Get the *Tobacco800 Document Image Database* from the official [source](http://tc11.cvc.uab.es/datasets/Tobacco800_1) and put all documents into the Tobacco800_Complete folder.

>All Jupyter Notebooks have a working_dir path parameter which needs to be set to the Tobacco800 folder.

## Notebooks
**Data_Preparation.ipybn**

Splits complete Tobacco800 into train and test dataset in the corresponding Tobacco800_Train and Tobacco800_Test folders. The train dataset is around 80% from the original Tobacco800 Dataset.

Generates resized and otsu binarized images from the complete Tobacco800 dataset into the Tobacco800_Small folder. The small version is used for the training and testing of the image model.

**CSV_Generation.ipybn**

Generates CSV Files needed for training and testing from the test and the train dataset with OCR to extract the text data from the corresponding files.

- tobacco800.train
- tobacco800.test

**TextModel_Training.ipybn**

Functionality to train two text models. The single page model takes the current page as the only input, the current & previous page model takes a pair of consecutive pages as input. The trained for both models is either 0 for continuity or 1 for rupture in the page stream.

**ImageModel_Training.ipybn**

Functionality to train two image models. The logic is the same as for the text models, with the difference being in the model architecture and the input being resized images.

**MultiModal_Combination.ipybn**

Functionality to combine one text and one image model in a late fusion approach to generate a multi-modal page stream segmentation prediction.

**BERT_Model_SinglePage.ipynb**

Functionality to train a single page text model based on the bert-base-uncased transformers model with the addition of combining it into a multi-modal page stream segmentation prediction with the image models from the other notebooks.