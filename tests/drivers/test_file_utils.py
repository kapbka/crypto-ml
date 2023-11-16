import unittest
from datetime import date

from common.storage.file import File


class FileNames(unittest.TestCase):
    def test_generate_names(self):
        self.assertEqual('btc_last_7days_1min_07august.csv', File(ts=date(2021, 8, 7)).name())
        self.assertEqual('btc_last_7days_1min_28august.csv', File(ts=date(2021, 8, 28)).name())
        self.assertEqual('btc_last_7days_1min_19september.csv', File(ts=date(2021, 9, 19)).name())

    def test_conversion(self):
        self.assertEqual(File.parse('btc_last_7days_1min_07august.csv').name(),
                         'btc_last_7days_1min_07august.csv')
        self.assertEqual(File.parse('btc_last_7days_1min_28august.csv').name(),
                         'btc_last_7days_1min_28august.csv')
        self.assertEqual(File.parse('btc_last_7days_1min_19september.csv').name(),
                         'btc_last_7days_1min_19september.csv')
