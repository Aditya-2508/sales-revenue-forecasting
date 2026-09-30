import numpy as np

from backend.ml.evaluation.metrics import (
    calculate_mae,
    calculate_rmse,
    calculate_wape,
)


def test_calculate_mae():
    actual = np.array([100.0, 200.0, 300.0])
    predicted = np.array([90.0, 220.0, 280.0])

    result = calculate_mae(actual, predicted)

    assert result == 16.666666666666668


def test_calculate_rmse():
    actual = np.array([100.0, 200.0, 300.0])
    predicted = np.array([90.0, 220.0, 280.0])

    result = calculate_rmse(actual, predicted)

    assert np.isclose(result, 17.32050807568877)


def test_calculate_wape():
    actual = np.array([100.0, 200.0, 300.0])
    predicted = np.array([90.0, 220.0, 280.0])

    result = calculate_wape(actual, predicted)

    assert np.isclose(result, 0.08333333333333333)


def test_calculate_wape_with_zero_actuals():
    actual = np.array([0.0, 0.0])
    predicted = np.array([10.0, 20.0])

    result = calculate_wape(actual, predicted)

    assert result == 0.0
    