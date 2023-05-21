# mle-template
Classic MLE template with CI/CD pipelines

Using technologies:
* Analytics and model training
    * Python 3.x
    * Pandas, NumPy, SkLearn
* Testing
  * unittest + coverage
* Data / Model versioning
    * DVC
* CI/CD
  * GitHub Actions

___

### Links: 
* [Docker Image]()

___

### Dataset

Twitter Sentiment Analysis Dataset from [Kaggle](https://www.kaggle.com/c/twitter-sentiment-analysis2).
Sentiment analysis is a common task in the field of Natural Language Processing (NLP). It is used to determine whether a piece of text is positive, negative, or neutral. In this dataset, the task is to classify the sentiment of tweets from Twitter.


___

### Workflow
1. Download dataset from [Kaggle](https://www.kaggle.com/c/twitter-sentiment-analysis2)
2. Analyze dataset and create simple baseline model in this [notebook](./notebooks/twitter-sentiment-analysis.ipynb)
3. Transform notebook to python scripts in [src](./src) folder
4. Put dataset into S3 bucket using DVC
5. Created Dockerfile and [docker-compose.yml](./docker-compose.yml)
6. Created CI / CD pipelines using GitHub Actions:
    * [CI](./.github/workflows/ci.yaml)
    * [CD](./.github/workflows/cd.yaml)

___

### Run tests

Run data preprocessing tests:
```bash
python -m unittest src/unit_tests/test_preprocess.py
```

Run model training tests:
```bash
python -m unittest src/unit_tests/test_training.py
```