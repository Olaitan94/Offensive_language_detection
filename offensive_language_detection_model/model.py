from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras.callbacks import ModelCheckpoint
from offensive_language_detection_model.config.core import config


#get the correct value for the initial bias
initial_bias = np.log([config.model_config.pos/config.model_config.neg])
output_bias = tf.keras.initializers.Constant(initial_bias)


def create_model():
    input_word_ids = tf.keras.Input(shape=(128,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(128,), dtype=tf.int32, name="input_mask")

    encoder = hub.KerasLayer(config.model_config.bert_model_url, trainable=True, name='BERT_encoder')

    encoder_output = encoder({'input_word_ids':input_word_ids, 'input_mask':input_mask},True)
    sequence_output = encoder_output['sequence_output']

    clf_output = sequence_output[:, 0, :]

    layer = layers.Dense(units = 64, activation = 'relu', name ='hidden_layer_1')(clf_output)
    layer = tf.keras.layers.Dropout(0.2)(layer)
    layer = tf.keras.layers.Dense(32, activation='relu', name = 'hidden_layer_2')(layer)
    out = tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)(layer)

    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

    return model

if __name__ == '__main__':
    model = create_model()
    model.summary()
