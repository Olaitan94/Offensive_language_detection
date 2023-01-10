import math

import numpy as np

from offensive_language_detection_model.predict import predict_text


def test_predict(sample_input_data):
    # Given
    expected_first_prediction_class = 'NORMAL'
    expected_first_prediction_prob = 0.00133703
    expected_no_predictions = 5

    # When
    result = sample_input_data.apply(predict_text)

    # Then
    first_val = result[0]
    first_val_text = first_val.get("ACTUAL TEXT")
    first_val_class = first_val.get("PREDICTED CLASS")
    first_val_prob = first_val.get("Probability")
    assert isinstance(first_val, dict)
    assert isinstance(first_val_text, str)
    assert isinstance(first_val_class, str)
    assert isinstance(first_val_prob, np.float32)
    assert first_val_class == expected_first_prediction_class
    assert math.isclose(first_val_prob, expected_first_prediction_prob, abs_tol=0.01)
    assert result.shape[0] == expected_no_predictions
