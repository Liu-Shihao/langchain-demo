#pip install langchain-openai
#export OPENAI_API_KEY="..."

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

chat = ChatOpenAI(openai_api_key="...") # Otherwise you can initialize without any params: chat = ChatOpenAI()

'''
The chat model interface is based around messages rather than raw text. 
The types of messages currently supported in LangChain are AIMessage, HumanMessage, SystemMessage, FunctionMessage and ChatMessage – ChatMessage takes in an arbitrary role parameter. 
Most of the time, you’ll just be dealing with HumanMessage, AIMessage, and SystemMessage
'''




messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(content="What is the purpose of model regularization?"),
]

chat.invoke(messages)

# AIMessage(content="The purpose of model regularization is to prevent overfitting in machine learning models. Overfitting occurs when a model becomes too complex and starts to fit the noise in the training data, leading to poor generalization on unseen data. Regularization techniques introduce additional constraints or penalties to the model's objective function, discouraging it from becoming overly complex and promoting simpler and more generalizable models. Regularization helps to strike a balance between fitting the training data well and avoiding overfitting, leading to better performance on new, unseen data.")

for chunk in chat.stream(messages):
    print(chunk.content, end="", flush=True)
# The purpose of model regularization is to prevent overfitting and improve the generalization of a machine learning model. Overfitting occurs when a model is too complex and learns the noise or random variations in the training data, which leads to poor performance on new, unseen data. Regularization techniques introduce additional constraints or penalties to the model's learning process, discouraging it from fitting the noise and reducing the complexity of the model. This helps to improve the model's ability to generalize well and make accurate predictions on unseen data.

chat.batch([messages])
# [AIMessage(content="The purpose of model regularization is to prevent overfitting in machine learning models. Overfitting occurs when a model becomes too complex and starts to learn the noise or random fluctuations in the training data, rather than the underlying patterns or relationships. Regularization techniques add a penalty term to the model's objective function, which discourages the model from becoming too complex and helps it generalize better to new, unseen data. This improves the model's ability to make accurate predictions on new data by reducing the variance and increasing the model's overall performance.")]

await chat.ainvoke(messages)
# AIMessage(content='The purpose of model regularization is to prevent overfitting in machine learning models. Overfitting occurs when a model becomes too complex and starts to memorize the training data instead of learning general patterns and relationships. This leads to poor performance on new, unseen data.\n\nRegularization techniques introduce additional constraints or penalties to the model during training, discouraging it from becoming overly complex. This helps to strike a balance between fitting the training data well and generalizing to new data. Regularization techniques can include adding a penalty term to the loss function, such as L1 or L2 regularization, or using techniques like dropout or early stopping. By regularizing the model, it encourages it to learn the most relevant features and reduce the impact of noise or outliers in the data.')

async for chunk in chat.astream(messages):
    print(chunk.content, end="", flush=True)
# The purpose of model regularization is to prevent overfitting in machine learning models. Overfitting occurs when a model becomes too complex and starts to memorize the training data instead of learning the underlying patterns. Regularization techniques help in reducing the complexity of the model by adding a penalty to the loss function. This penalty encourages the model to have smaller weights or fewer features, making it more generalized and less prone to overfitting. The goal is to find the right balance between fitting the training data well and being able to generalize well to unseen data.