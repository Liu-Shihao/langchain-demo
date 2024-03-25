from langchain import hub
from langchain_community.llms.ollama import Ollama
from langchain_core.runnables import RunnablePassthrough, RunnablePick

rag_prompt = hub.pull("rlm/rag-prompt")
rag_prompt.messages


llm = Ollama(model="llama2")


# Chain
chain = (
    RunnablePassthrough.assign(context=RunnablePick("context") | format_docs)
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Run
chain.invoke({"context": docs, "question": question})
