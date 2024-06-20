import spacy
from spacy.training import Example
from spacy.util import load_config
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import copy
import warnings

# 忽略不重要的警告
warnings.filterwarnings("ignore", category=UserWarning, module='spacy')


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

    def fit(self, X, y=None):
        config = copy.deepcopy(self.config)

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

        self.nlp = spacy.blank(config['nlp']['lang'])
        textcat = self.nlp.add_pipe('textcat')
        textcat.add_label('A')
        textcat.add_label('B')
        textcat.add_label('C')
        textcat.add_label('D')

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
        return [max(self.nlp(text).cats, key=self.nlp(text).cats.get) for text in X]


# 示例训练数据
train_data = [
    ("Text of example A", {"cats": {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0}}),
    ("Text of example B", {"cats": {"A": 0.0, "B": 1.0, "C": 0.0, "D": 0.0}}),
    # Add more examples as needed
]

# 超参数网格
param_grid = {
    'dropout': [0.1, 0.5],
    'patience': [10, 50],
    'learn_rate': [0.0001, 0.01],
    'max_epochs': [10, 50],
    'max_steps': [1000, 5000],
}

# 配置文件路径
config_path = "config.cfg"
spacy_classifier = SpacyClassifier(config_path)

# K折交叉验证
kf = KFold(n_splits=5)
grid_search = GridSearchCV(estimator=spacy_classifier, param_grid=param_grid, scoring='accuracy', cv=kf, verbose=1,
                           n_jobs=-1)

# 训练模型并进行参数调优
grid_search.fit(train_data)

print("Best parameters found: ", grid_search.best_params_)
print("Best score found: ", grid_search.best_score_)
