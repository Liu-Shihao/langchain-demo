from langchain_community.llms import Ollama
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

"""
https://python.langchain.com/docs/guides/local_llms#quickstart
https://python.langchain.com/docs/guides/local_llms#ollama
https://python.langchain.com/docs/integrations/llms/ollama

Ollama and llamafile will automatically utilize the GPU on Apple devices.

在 Mac 上，模型将下载到~/.ollama/models
在 Linux（或 WSL）上，模型将存储在 /usr/share/ollama/.ollama/models
"""
# llm = Ollama(model="llama2")
# resp = llm("The first man on the moon was ...")

# 流式输出
llm = Ollama(
    model="llama2", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)
resp = llm("The first man on the moon was ...")

# print(resp)