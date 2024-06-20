import spacy
from sklearn.base import BaseEstimator
from spacy.training import Example
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, make_scorer, accuracy_score
import numpy as np

# 准备数据（假设你已经准备好了数据）
train_data = [
    ("I want transfer to Ajay $100 Tomorrow", {"cats": {"TRANSFER": 1.0, "SPORTS": 0.0, "FOOD": 0.0, "TRAVEL": 0.0}}),
    ("I want to play football", {"cats": {"TRANSFER": 0.0, "SPORTS": 1.0, "FOOD": 0.0, "TRAVEL": 0.0}}),
    ("I want to eat watermelon.", {"cats": {"TRANSFER": 0.0, "SPORTS": 0.0, "FOOD": 1.0, "TRAVEL": 0.0}}),
    ("I want to go to Malaysia", {"cats": {"TRANSFER": 0.0, "SPORTS": 0.0, "FOOD": 0.0, "TRAVEL": 1.0}}),
    # Add more examples as needed
]

# 定义spaCy的train config模板
config_template = {
    "nlp": {
        "textcat": {
            "architecture": "simple_cnn",
            "threshold": 0.5,
        }
    },
    "training": {
        "batcher": {
            "discard_oversize": True,
            "size": 2000,
            "get_length": None,
        },
        "optimizer": {
            "learn_rate": 0.001,  # 默认的学习率
        },
        "patience": 100,  # 默认的耐心值
        "max_epochs": 20,  # 默认的最大迭代次数
        "eval_frequency": 1000,
    }
}

# 设置参数网格
param_grid = {
    'training.batcher.size': [1000, 2000, 3000],
    'training.patience': [50, 100, 200],
    'training.optimizer.learn_rate': [0.001, 0.01, 0.1],
    'training.max_epochs': [10, 20, 30],
}

# 设置评估指标
def custom_scorer(y_true, y_pred):
    # 假设这里使用准确率作为评估指标
    return accuracy_score(y_true, y_pred)

# 实例化spaCy的blank模型
nlp = spacy.blank('en')
textcat = nlp.add_pipe('textcat')
textcat.add_label('TRANSFER')
textcat.add_label('SPORTS')
textcat.add_label('FOOD')
textcat.add_label('TRAVEL')


# 定义一个包装器类，使spaCy模型能够与GridSearchCV兼容
class SpacyClassifier(BaseEstimator):
    def __init__(self, config_template):
        self.config_template = config_template
        self.nlp = spacy.blank('en')
        self.textcat = self.nlp.add_pipe('textcat')
        self.textcat.add_label('TRANSFER')
        self.textcat.add_label('SPORTS')
        self.textcat.add_label('FOOD')
        self.textcat.add_label('TRAVEL')

    def fit(self, X, y=None):
        # 转换数据格式
        train_texts = [item[0] for item in X]
        train_cats = [item[1] for item in X]

        train_examples = []
        for text, cats in zip(train_texts, train_cats):
            train_examples.append(Example.from_dict(self.nlp.make_doc(text), cats))

        # 初始化模型并训练
        self.nlp.initialize(lambda: train_examples)
        self.nlp.train(train_examples, None, config=self.config_template)

    def predict(self, X):
        # 在输入数据上进行预测
        return [max(self.nlp(text).cats, key=self.nlp(text).cats.get) for text in X]


# 创建一个SpacyClassifier对象作为estimator
spacy_classifier = SpacyClassifier(config_template)

# 创建GridSearchCV对象
grid_search = GridSearchCV(estimator=spacy_classifier, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)

# 执行网格搜索
grid_search.fit(train_data)

# 输出最佳参数和评估结果
print("Best parameters found: ", grid_search.best_params_)
print("Best score found: ", grid_search.best_score_)
