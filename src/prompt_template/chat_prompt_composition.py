from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

prompt = SystemMessage(content="You are a nice pirate")

new_prompt = (
    prompt + HumanMessage(content="hi") + AIMessage(content="what?") + "{input}"
)



print(new_prompt.format_messages(input="i said hi"))
# [SystemMessage(content='You are a nice pirate'),
# HumanMessage(content='hi'),
# AIMessage(content='what?'),
# HumanMessage(content='i said hi')]