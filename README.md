**Intro**
This package contains the offensive_language_detection_model, which can be used to detect offensive language in a given text.

**Config**
This module contains the file (core.py) which is used to set the configuration for all the variables & file paths needed to run the package. The variables are listed in the config.yml file

**Datasets**
offensive_language_detection_model.datasets - This module has the original datasets which were used to train & test the model
  - test_data.csv
  - train_data.csv

**Processing**
offensive_language_detection_model.processing contains the preprocessing module.
This contains functions necessary to transform raw data into the format expected by the model. Hence, this module is the preprocessing pipeline for this model.

**trained_models**
offensive_language_detection_model.trained_models contains the latest version of the trained model.h5 file

**config.yml**
Contains names of all the variables & file paths used in the model

**model.py**
contains the code for creating an instance of the model

**predict.py**
contains code for detecting if a text is offensive

**train_model.py**
This file can be run to retrain the model and generate a new model.h5 file

**Requirements**
contains dependencies required to use or test the package

**Tests**
contains scripts for testing the package

**conftest.py** - contains a fixture fxn which is used to provide the test data to the other test file.
**test_prediction.py** - used to test the predict_text function of the offensive_language_detection_model

**Manifest.in**
contains instructions for what to include or exclude when building the package

**pyproject.toml**
this file contains basic dependencies for setting up the package and also the configuration options for pytest.
