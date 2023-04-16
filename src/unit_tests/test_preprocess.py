import configparser
import os
import sys
import unittest

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

config = configparser.ConfigParser()
config.read("config.ini")


class TestPreprocess(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
