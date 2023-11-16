import numpy as np
from datetime import datetime
from pandas import DataFrame, Timestamp
from tempfile import NamedTemporaryFile

from common.plot_tools import plot_columns, plot_to_file, interactive_plot

df = DataFrame({c: np.random.randint(0, 1, 100) for c in ['close', '2', '3']},
               index=np.array([datetime.fromtimestamp(i * 60) for i in range(100)]))


def test_plot():
    plot_columns(df, ['close', '2', ['2', '3']], text=[(Timestamp(datetime.fromtimestamp(0)), "test")])


def test_interactive_plot():
    interactive_plot(df, ['close', '2', '3'])


def test_plot_to_file():
    with NamedTemporaryFile('w') as temp_file:
        plot_to_file(df, ['2', '3'], temp_file.name, 10, 10)
