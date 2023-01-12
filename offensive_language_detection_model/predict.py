from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from offensive_language_detection_model.config.core import config,TRAINED_MODEL_DIR
from offensive_language_detection_model.processing.preprocessing import text_processing_pipeline, get_bert_token


def predict_text(text: str) -> dict:
    """Make a prediction using a saved model."""

    text = text
    probability = None
    predicted_class = None

    #clean the text
    processed_text = text_processing_pipeline(text)
    #get the tokens for the text
    text_token = get_bert_token([processed_text])
    #load the saved model
    saved_model_path = TRAINED_MODEL_DIR
    loaded_model = tf.keras.models.load_model(saved_model_path, custom_objects={'KerasLayer':hub.KerasLayer})
    #Predict the text
    probability = loaded_model.predict(text_token)[0][0]
    threshold = config.model_config.threshold
    if probability >= threshold:
        predicted_class = 'OFFENSIVE'
    else:
        predicted_class = 'NORMAL'

    results = {"ACTUAL_TEXT": text, "PREDICTED_CLASS": predicted_class,"Probability": probability}

    return results
