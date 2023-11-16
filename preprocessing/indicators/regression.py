import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from preprocessing.indicators.base import Indicator


class Regression(Indicator):
    def __init__(self, window: int, degree: int = 2):
        super().__init__(window=window)
        self._degree = degree
        self._x_buffer = np.array(range(window)).reshape((window, 1))

        self._poly_reg = PolynomialFeatures(degree=self._degree)
        self._x_poly = self._poly_reg.fit_transform(self._x_buffer)

    def process(self, previous: float) -> float:
        # check if first value in the buffer is nan to avoid exceptions
        if np.isnan(self._buffer[0]):
            return np.nan

        lin_reg_2 = LinearRegression()
        lin_reg_2.fit(self._x_poly, self._buffer)

        return lin_reg_2.predict(self._poly_reg.fit_transform([[self._window]]))[0]


