import configparser
import os
import sys
import unittest

from preprocess import clean_text, split_for_validation

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

config = configparser.ConfigParser()
config.read("config.ini")


class TestPreprocess(unittest.TestCase):
    def setUp(self) -> None:
        self.trainer = Trainer.default_trainer(config['unit_testing']['train_path'],
                                               config['unit_testing']['test_path'])

    def test_get_train_data(self):
        self.assertTrue(type(self.trainer.get_train()) is DataFrame)

    def test_get_test_data(self):
        self.assertTrue(type(self.trainer.get_test()) is DataFrame)


if __name__ == "__main__":
    unittest.main()
