#This file contains code that will be used to test the config settings in config.py & config.yml

import pytest
import pandas as pd
from pathlib import Path
from offensive_language_detection_model.config.core import config, DATASET_DIR
from offensive_language_detection_model.processing.preprocessing import text_processing_pipeline, get_bert_token

#This fixture will load the test data set which will be used by the test_prediction.py file
@pytest.fixture()
def sample_input_data():
    file_name = config.app_config.test_data_file
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    series = dataframe['tweet'][:5]
    return series
