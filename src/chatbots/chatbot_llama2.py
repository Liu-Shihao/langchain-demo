from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory import ChatMessageHistory
"""
https://python.langchain.com/docs/use_cases/chatbots/quickstart
https://python.langchain.com/docs/integrations/llms/ollama#usage
"""

chat = ChatOllama(odel="llama2")

prompt = ChatPromptTemplate.from_messages(
    [
        # (
        #     "system",
        #     "You are a helpful assistant. Answer all questions to the best of your ability.",
        # ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | chat

# 管理聊天历史记录,它负责保存和加载聊天消息。有许多内置消息历史记录集成可将消息保存到各种数据库
demo_ephemeral_chat_history = ChatMessageHistory()

demo_ephemeral_chat_history.add_user_message(
    "Translate this sentence from English to French: I love programming."
)

response = chain.invoke({"messages": demo_ephemeral_chat_history.messages})

demo_ephemeral_chat_history.add_ai_message(response)


demo_ephemeral_chat_history.add_user_message("What did you just say?")


# resp = chain.invoke(
#     {
#         "messages": [
#             HumanMessage(
#                 content="Translate this sentence from English to French: I love programming."
#             ),
#             AIMessage(content="J'adore la programmation."),
#             HumanMessage(content="What did you just say?"),
#         ],
#     }
# )

resp = chain.invoke({"messages": demo_ephemeral_chat_history.messages})

print(resp)
# content='\nIn French, "J\'aime le programme" means "I like programming."'