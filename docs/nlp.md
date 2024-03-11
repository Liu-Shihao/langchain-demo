# NLP工具
以下是一些流行的开源NLP工具：

1. **NLTK (Natural Language Toolkit)**：NLTK是Python的一个库，提供了各种NLP任务的模块，包括分词、词性标注、命名实体识别、句法分析等。

2. **spaCy**：spaCy是另一个流行的Python NLP库，提供了快速的实体识别、句法分析等功能，并且优化了性能。

3. **Gensim**：Gensim是一个用于主题建模和词向量表示的Python库，可以用于文本相似度计算、主题建模、文档摘要等任务。

4. **Stanford CoreNLP**：Stanford CoreNLP是斯坦福大学开发的一个NLP工具包，提供了丰富的功能，包括分词、词性标注、句法分析、命名实体识别等。

5. **BERT (Bidirectional Encoder Representations from Transformers)**：BERT是谷歌发布的预训练语言模型，可以用于各种NLP任务，如文本分类、命名实体识别、问答等。

6. **Transformers**：Hugging Face开发的Transformers库提供了各种预训练的Transformer模型，包括BERT、GPT、RoBERTa等，可以用于各种NLP任务。

7. **AllenNLP**：AllenNLP是由艾伦人工智能研究所开发的一个用于深度学习在NLP任务上的库，提供了各种模型和工具，包括文本分类、命名实体识别、句法分析等。

8. **fastText**：fastText是Facebook开发的一个用于文本分类和词向量表示的库，可以用于各种NLP任务。

这些工具提供了丰富的功能和API，可以用于各种NLP任务，你可以根据具体的需求选择适合的工具。

## Install spaCy
macOs ARM/M1 Cpu  
```shell
pip install -U pip setuptools wheel
pip install -U 'spacy[apple]'
python -m spacy download en_core_web_sm

#Installing collected packages: en-core-web-sm
#Successfully installed en-core-web-sm-3.7.1
#✔ Download and installation successful
#You can now load the package via spacy.load('en_core_web_sm')
```
