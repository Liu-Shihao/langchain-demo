import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import numpy as np


# 自定义模型训练函数
def train_spacy_model(nlp, train_data, batch_size=16, dropout=0.1, learn_rate=0.001, max_epochs=10):
    optimizer = nlp.begin_training()
    optimizer.learn_rate = learn_rate
    for epoch in range(max_epochs):
        losses = {}
        batches = minibatch(train_data, size=compounding(4.0, batch_size, 1.001))
        for batch in batches:
            examples = []
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)
            nlp.update(examples, sgd=optimizer, drop=dropout, losses=losses)
        print(f'Epoch {epoch}, Loss: {losses}')
    return nlp


# 自定义评分函数
def custom_scorer(estimator, X, y, **kwargs):
    examples = [Example.from_dict(estimator.nlp.make_doc(text), ann) for text, ann in zip(X, y)]
    scores = estimator.nlp.evaluate(examples)
    return scores['ents_f']


# 包装成scikit-learn兼容的函数
class SpacyEstimator:
    def __init__(self, nlp=None, batch_size=16, dropout=0.1, learn_rate=0.001, max_epochs=10):
        if nlp is None:
            nlp = spacy.blank("en")
        self.nlp = nlp
        self.batch_size = batch_size
        self.dropout = dropout
        self.learn_rate = learn_rate
        self.max_epochs = max_epochs

    def fit(self, X, y):
        self.nlp = train_spacy_model(self.nlp, list(zip(X, y)), batch_size=self.batch_size, dropout=self.dropout,
                                     learn_rate=self.learn_rate, max_epochs=self.max_epochs)
        return self

    def score(self, X, y):
        return custom_scorer(self, X, y)

    def predict(self, X):
        docs = [self.nlp(text) for text in X]
        return docs

    def get_params(self, deep=True):
        return {"nlp": self.nlp, "batch_size": self.batch_size, "dropout": self.dropout, "learn_rate": self.learn_rate,
                "max_epochs": self.max_epochs}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


train_data = [
    # [startInd,endInd)
    ("Apple is looking at buying U.K. startup for $1 billion", {"entities": [(0, 5, "ORG"), (27, 31, "GPE"), (44, 54, "MONEY")]}),
    ("San Francisco considers banning sidewalk delivery robots", {"entities": [(0, 13, "GPE")]}),
    ("London is a big city in the United Kingdom.", {"entities": [(0, 6, "GPE"), (28, 43, "GPE")]}),
    ("Tesla is planning to build a new factory in Berlin", {"entities": [(0, 5, "ORG"), (44, 50, "GPE")]}),
    ("Amazon has announced new headquarters in New York", {"entities": [(0, 6, "ORG"), (41, 49, "GPE")]}),
    ("Google's CEO Sundar Pichai introduced the new Pixel at Google I/O.", {"entities": [(0, 6, "ORG"), (13, 26, "PERSON")]})
]


# 准备数据
X = [text for text, _ in train_data]
y = [ann for _, ann in train_data]

# 加载空白模型
nlp = spacy.blank("en")

# 定义参数网格
param_grid = {
    'batch_size': [16, 32],
    'dropout': [0.1, 0.2],
    'learn_rate': [0.001, 0.0001],
    'max_epochs': [10, 20]
}

# 使用GridSearchCV进行超参数调优
grid_search = GridSearchCV(estimator=SpacyEstimator(nlp=nlp), param_grid=param_grid, scoring=make_scorer(custom_scorer),
                           cv=3)
grid_search.fit(X, y)

print("Best parameters found: ", grid_search.best_params_)
print("Best score found: ", grid_search.best_score_)
