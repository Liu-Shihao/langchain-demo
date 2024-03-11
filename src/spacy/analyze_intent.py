import spacy

# 加载英文模型
nlp = spacy.load("en_core_web_sm")


# 定义一个函数来分析用户输入的搜索查询
def analyze_intent(query):
    # 对用户查询进行处理
    doc = nlp(query)

    # 获取查询中的动词
    verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]

    # 获取查询中的名词短语
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]

    # 获取查询中的命名实体
    entities = [ent.text for ent in doc.ents]

    return verbs, noun_phrases, entities


# 测试函数
query = "I want to buy a new laptop for gaming"
verbs, noun_phrases, entities = analyze_intent(query)
print("动词:", verbs)
print("名词短语:", noun_phrases)
print("命名实体:", entities)
'''
动词: ['want', 'buy', 'game']
名词短语: ['I', 'a new laptop']
命名实体: []
'''
