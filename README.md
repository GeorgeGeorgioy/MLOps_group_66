# MLOps Group 66
Overall goal of the project:
Credit card fraud is increasingly common in the modern world of digitalized commerce. See https://merchantcostconsulting.com/lower-credit-card-processing-fees/credit-card-fraud-statistics/

Modern problems require modern solutions, and therefore the potential of automated fraud detection is still widely developed. In this project we want to implement a credit card fraud detection neural network, to learn more about the topic and develop a robust pipleine for furhter model development and operations.

What framework are you going to use, and you do you intend to include the framework into your project?
A pretrained transformer based tabular model from huggingface, built in tensorflow:
https://huggingface.co/keras-io/tab_transformer
The model is trained on United States Census Income Dataset provided by the UC Irvine Machine Learning Repository.


What data are you going to run on (initially, may change)
Credit card fraud detection dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud'
The dataset consists of a time indicator for when each transaction was performed, as well the transactional amount in each purchase as well as a binary label of whether or not the case was fradulent. In addition to these, 28 principal components are included, which have been processed to avoid sharing personal sensitive information. These components will serve as the primary classification features used to classify whether or not a purchase is fradulent.

What models do you expect to use
We expect first use a fully connected artificial neural network, that will help us set up a basic pipeline for the project. Furthermore, we will attempt to use the TabTransformer from the huggingface framework, to improve model performance and test a transformer on the credit card fraud detection dataset, if time allows.


## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
