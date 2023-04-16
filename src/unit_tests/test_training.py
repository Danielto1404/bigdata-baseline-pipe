import configparser
import os
import sys
import unittest
import pandas as pd
from train import TweetsClassificationTrainer

sys.path.insert(1, os.path.join(os.getcwd(), "src"))


config = configparser.ConfigParser()
config.read("config.ini")


class TestTraining(unittest.TestCase):
    def setUp(self) -> None:
        self.trainer = TweetsClassificationTrainer.default_trainer(config["data"]["train"],
                                                                   config["data"]["test"])

    def test_get_train_data(self):
        self.assertTrue(type(self.trainer.get_train()) is pd.DataFrame)

    def test_get_test_data(self):
        self.assertTrue(type(self.trainer.get_test()) is pd.DataFrame)

    def test_train_columns_preprocessed(self):
        self.assertTrue("tweet" in self.trainer.get_train().columns)
        self.assertTrue("label" in self.trainer.get_train().columns)

    def test_test_columns_preprocessed(self):
        self.assertTrue("tweet" in self.trainer.get_test().columns)
        self.assertTrue("label" in self.trainer.get_test().columns)


if __name__ == "__main__":
    unittest.main()
