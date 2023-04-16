import configparser
import os
import sys
import unittest
import pandas as pd

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from train import TweetsClassificationTrainer
from preprocess import clean_text, split_for_validation

config = configparser.ConfigParser()
config.read("config.ini")


class TestPreprocess(unittest.TestCase):
    def setUp(self) -> None:
        self.trainer = TweetsClassificationTrainer.default_trainer(
            config["data"]["train"],
            config["data"]["test"]
        )

        self.train_path = config["data"]["train"]
        self.test_path = config["data"]["test"]

    def test_clean_text_simple(self):
        text = "This is a test text"
        self.assertEqual(clean_text(text), "this is test text")

    def test_clean_text_complex(self):
        text = "This is a test text with 123 numbers and https://www.example.com url"
        self.assertEqual(clean_text(text), "this is test text with numbers and url")

    def test_clean_text_empty(self):
        text = ""
        self.assertEqual(clean_text(text), "")

    def test_train_columns_preprocessed(self):
        self.assertTrue("id" in self.trainer.get_train().columns)
        self.assertTrue("tweet" in self.trainer.get_train().columns)
        self.assertTrue("label" in self.trainer.get_train().columns)

    def test_test_columns_preprocessed(self):
        self.assertTrue("id" in self.trainer.get_train().columns)
        self.assertTrue("tweet" in self.trainer.get_test().columns)
        self.assertTrue("label" in self.trainer.get_test().columns)

    def test_split_for_validation(self):
        train_df = pd.read_csv(self.train_path)
        x_train, y_train = train_df["tweet"], train_df["label"]
        x_train = x_train.apply(clean_text)
        x_train, x_val, y_train, y_val = split_for_validation(train_df)
        self.assertTrue(len(x_train) > 0)
        self.assertTrue(len(x_val) > 0)
        self.assertTrue(len(y_train) > 0)
        self.assertTrue(len(y_val) > 0)


if __name__ == "__main__":
    unittest.main()
