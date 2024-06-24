import spacy
from spacy.training import Example
from spacy.util import minibatch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score
import numpy as np

class SpacyTextCatClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, nlp, max_epochs=10, max_steps=None, batch_size=32, dropout=0.2, learn_rate=0.001):
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.learn_rate = learn_rate
        self.nlp = spacy.blank('en')
        self.textcat = self.nlp.add_pipe('textcat', last=True)
        self.textcat.add_label("POSITIVE")
        self.textcat.add_label("NEGATIVE")
        self.classes_ = np.array(["NEGATIVE", "POSITIVE"])

    def fit(self, X, y):
        train_data = [(text, {"cats": {"POSITIVE": label, "NEGATIVE": not label}}) for text, label in zip(X, y)]
        optimizer = self.nlp.begin_training()
        optimizer.learn_rate = self.learn_rate
        for epoch in range(self.max_epochs):
            losses = {}
            batches = minibatch(train_data, size=self.batch_size)
            for batch in batches:
                texts, annotations = zip(*batch)
                examples = [Example.from_dict(self.nlp.make_doc(text), annotation) for text, annotation in zip(texts, annotations)]
                self.nlp.update(examples, drop=self.dropout, sgd=optimizer, losses=losses)
        return self

    def predict(self, X):
        return [self._predict_single(text) for text in X]

    def _predict_single(self, text):
        doc = self.nlp(text)
        scores = doc.cats
        return max(scores, key=scores.get) == "POSITIVE"

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

# 准备数据
X_train = ["I love this movie", "I hate this movie", "This movie is great", "This movie is terrible"]
y_train = [1, 0, 1, 0]

# 定义参数网格
param_grid = {
    'classifier__dropout': [0.2, 0.5],
    'classifier__max_epochs': [5, 10],
    'classifier__batch_size': [4, 8],
    'classifier__learn_rate': [0.001, 0.01, 0.1]
}

# 创建模型管道
pipe = Pipeline([
    ('classifier', SpacyTextCatClassifier())
])

# 使用GridSearchCV进行参数调优
grid_search = GridSearchCV(pipe, param_grid, cv=2, scoring=make_scorer(accuracy_score))
grid_search.fit(X_train, y_train)

# 输出最佳参数和分数
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)
