
import spacy
from spacy.training import Example
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut
import copy

from spacy.util import load_config


# 定义一个包装器类，使spaCy模型能够与GridSearchCV兼容
class SpacyClassifier(BaseEstimator):
    def __init__(self, config_path, dropout=None, patience=None, learn_rate=None, max_epochs=None, max_steps=None):
        self.config_path = config_path
        self.dropout = dropout
        self.patience = patience
        self.learn_rate = learn_rate
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.config = load_config(config_path)
        self.nlp = None

    def fit(self, X, y=None, **params):
        # 复制配置文件内容
        config = copy.deepcopy(self.config)

        # 解析额外的参数并更新到配置中
        if self.dropout is not None:
            config['training']['dropout'] = self.dropout
        if self.patience is not None:
            config['training']['patience'] = self.patience
        if self.learn_rate is not None:
            config['training']['optimizer']['learn_rate'] = self.learn_rate
        if self.max_epochs is not None:
            config['training']['max_epochs'] = self.max_epochs
        if self.max_steps is not None:
            config['training']['max_steps'] = self.max_steps

        # 初始化spaCy模型
        self.nlp = spacy.blank(config['nlp']['lang'])
        textcat = self.nlp.add_pipe('textcat')
        textcat.add_label('TRANSFER')
        textcat.add_label('SPORTS')
        textcat.add_label('FOOD')
        textcat.add_label('TRAVEL')

        # 转换数据格式
        train_texts = [item[0] for item in X]
        train_cats = [item[1] for item in X]

        train_examples = []
        for text, cats in zip(train_texts, train_cats):
            train_examples.append(Example.from_dict(self.nlp.make_doc(text), cats))

        # 初始化模型
        self.nlp.initialize(lambda: train_examples)

        # 使用update方法进行训练
        for epoch in range(self.max_epochs if self.max_epochs else 20):
            losses = {}
            self.nlp.update(train_examples, drop=self.dropout if self.dropout else 0.1, losses=losses)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {losses['textcat']}")

    def predict(self, X):
        # 在输入数据上进行预测
        return [max(self.nlp(text).cats, key=self.nlp(text).cats.get) for text in X]


# 准备数据（假设你已经准备好了数据）
train_data = [
    ("I want transfer to Ajay $100 Tomorrow", {"cats": {"TRANSFER": 1.0, "SPORTS": 0.0, "FOOD": 0.0, "TRAVEL": 0.0}}),
    ("I want to play football", {"cats": {"TRANSFER": 0.0, "SPORTS": 1.0, "FOOD": 0.0, "TRAVEL": 0.0}}),
    ("I want to eat watermelon.", {"cats": {"TRANSFER": 0.0, "SPORTS": 0.0, "FOOD": 1.0, "TRAVEL": 0.0}}),
    ("I want to go to Malaysia", {"cats": {"TRANSFER": 0.0, "SPORTS": 0.0, "FOOD": 0.0, "TRAVEL": 1.0}}),
    # Add more examples as needed
]



# 设置参数网格
"""
training.dropout: 这是用于防止模型过拟合的一种正则化技术。它指定了在训练过程中丢弃神经元的比例。通常范围在 0.0 到 0.5 之间。例如：[0.1, 0.2, 0.3, 0.4, 0.5]。
training.patience: 这是用于早停的参数，表示在验证集性能不再提升时，允许训练继续进行的最大次数。它的值越大，模型越有耐心等待更好的性能。通常范围在 10 到 100 之间。例如：[10, 20, 50, 100]。
training.optimizer.learn_rate: 这是优化器的学习率，决定了模型参数在每次迭代中更新的步长。范围可以从 0.0001 到 0.1。例如：[0.0001, 0.001, 0.01, 0.1]。
training.max_epochs: 这是训练的最大迭代次数。它决定了训练的总轮数。范围通常在 10 到 100 之间。例如：[10, 20, 30, 50, 100]。
training.max_steps: 这是训练的最大步数（迭代次数）。当设置了 max_steps 时，即使 max_epochs 还没有达到，训练也会提前停止。范围可以根据数据集大小和模型复杂度决定。通常可以设置较大的值，例如：[1000, 5000, 10000, 20000]。
"""
# param_grid = {
#     'training.dropout': [0.1, 0.2, 0.3, 0.4],
#     'training.patience': [10, 20, 50, 100],
#     'training.optimizer.learn_rate': [0.0001, 0.001, 0.01, 0.1],
#     'training.max_epochs': [10, 20, 30, 50, 100],
#     'training.max_steps': [1000, 5000, 10000, 20000],
# }

param_grid = {
    'dropout': [0.1, 0.5],
    'patience': [10, 50],
    'learn_rate': [0.0001, 0.01],
    'max_epochs': [10, 50],
    'max_steps': [1000, 5000],
}

config_path = "config.cfg"
# 创建一个SpacyClassifier对象作为estimator
spacy_classifier = SpacyClassifier(config_path)

# 创建GridSearchCV对象，使用Leave-One-Out交叉验证
loo = LeaveOneOut()
grid_search = GridSearchCV(estimator=spacy_classifier, param_grid=param_grid, scoring='accuracy', cv=loo, verbose=1)
# grid_search = GridSearchCV(estimator=spacy_classifier, param_grid=param_grid, scoring='accuracy', cv=4, verbose=1, n_jobs=-1)


# 执行网格搜索
grid_search.fit(train_data)

# 输出最佳参数和评估结果
print("Best parameters found: ", grid_search.best_params_)
print("Best score found: ", grid_search.best_score_)
