import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras.callbacks import ModelCheckpoint
from offensive_language_detection_model.config.core import config, TRAINED_MODEL_DIR, DATASET_DIR
from offensive_language_detection_model.model import create_model
from offensive_language_detection_model.processing.preprocessing import text_processing_pipeline, get_bert_token


def run_training() -> None:
    """Train the model."""

    #get the path of the train set
    train_data_path = DATASET_DIR / 'train_data.csv'

    #Create variable for the path to store the trained model
    save_model_path = TRAINED_MODEL_DIR

    #create callback function for model
    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_model_path, monitor='val_accuracy', save_best_only=True, verbose=1)

    #Epoch for training
    epochs = config.model_config.epochs

    #Validation split for training
    validation_split = config.model_config.validation_split

    #Class weights for each class
    weight_for_0 = (1/config.model_config.neg) * (config.model_config.total_rows/2)
    weight_for_1 = (1/config.model_config.pos) * (config.model_config.total_rows/2)

    #create a dictionary to hold the weights
    class_weights = {0:weight_for_0, 1:weight_for_1 }

    #Load the train datasets
    train_set = pd.read_csv(train_data_path)

    #clean the texts
    cleaned_texts = train_set['tweet'].apply(text_processing_pipeline)

    #convert the text into tokens for the bert model
    tokens = get_bert_token(cleaned_texts)

    #create the model
    classifier_model = create_model()
    history = classifier_model.fit(tokens,train_set['label'], epochs = epochs,validation_split=validation_split, callbacks=[checkpoint], verbose=1, class_weight=class_weights)

if __name__ == "__main__":
    run_training()
