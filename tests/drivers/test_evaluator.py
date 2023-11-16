import numpy as np

from common.constants import EvaluationMethod
from common.metrics.evaluator import evaluate

PRICES = np.array([
    np.array([100, 101, 99]),
    np.array([110, 111, 109]),
    np.array([120, 121, 119]),
    np.array([100, 101, 99]),
    np.array([100, 101, 99]),
    np.array([130, 131, 129]),
])


def test_eval_high_low_lo():
    predictions = np.array([
        np.array([0.001, 0, 0]),  # minimal buy limit order that should be executed on next step
        np.array([0, 0, 0]),      # do nothing
        np.array([0, 0.001, 0]),  # minimal sell limit order that should be executed on next step
        np.array([0, 0, 0]),      # do nothing
        np.array([11, 0, 0]),     # immediate buy
        np.array([0, 11, 0]),     # immediate sell
    ])
    money = np.zeros(len(predictions))
    res = evaluate(PRICES, predictions, money, EvaluationMethod.HighLowLO, 0, 0)

    assert res == ((0.5356963958478264, 0.5356963958478264, 2),
                   (0.0, 0.0, 0.0, 0),
                   [(1, 100.9999, 3, 119.790121, 118.36710783686023, 0.0, 18.367107836860228),
                    (4, 100.0, 5, 130.0, 153.56963958478264, 0.0, 53.569639584782635)])
    assert money[-1] == 153.56963958478264


def test_eval_capped_lo():
    predictions = np.array([
        np.array([0.0001, 0, 0]),  # big buy limit order that should be adjusted and executed on next step
        np.array([0, 0, 0]),       # do nothing
        np.array([0, 0.0001, 0]),  # big sell limit order that should be executed on next step
        np.array([0, 0, 0]),       # do nothing
        np.array([11, 0, 0]),      # immediate buy
        np.array([0, 11, 0]),      # immediate sell
    ])
    cap = 0.001

    money = np.zeros(len(predictions))
    res = evaluate(PRICES, predictions, money, EvaluationMethod.CappedLOPrice, 0, 0, cap, cap, False)

    assert res == ((0.5506649194883102, 0.5506649194883102, 2),
                   (0.0, 0.0, 0.0, 0),
                   [(1, 100.1, 3, 119.88, 119.52083904095905, 0.0, 19.520839040959045),
                    (4, 100.0, 5, 130.0, 155.06649194883101, 0.0, 55.066491948831015)])
    assert money[-1] == 155.06649194883101


def test_eval_oco():
    predictions = np.array([
        np.array([0.0001, 0, 0]),  # big buy limit order that should be adjusted and executed on next step
        np.array([0, 0, 0]),       # do nothing
        np.array([0, 0.0001, 0]),  # big sell limit order that should be executed on next step
        np.array([0, 0, 0]),       # do nothing
        np.array([11, 0, 0]),      # immediate buy
        np.array([0, 11, 0]),      # immediate sell
    ])
    limit = 0.001
    rise_coeff = 0.001

    money = np.zeros(len(predictions))
    res = evaluate(PRICES, predictions, money, EvaluationMethod.OCO, 0, 0, 0, 0, False, limit, limit, rise_coeff)

    assert res == ((0.29610389870000003, 0.29610389870000003, 2),
                   (0, 0.0, 130.13, 129.87),
                   [(0, 100.0, 1, 100.1, 99.8999001, 0.0, -0.10009990000000357),
                    (5, 100.1, 0, 0.0, 0.0, 0.997002999, -0.19999980010000228)])
    assert money[-1] == 129.61038987
