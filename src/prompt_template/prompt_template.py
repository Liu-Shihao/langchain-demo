from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.prompt_values import StringPromptValue
from langchain_core.prompts import HumanMessagePromptTemplate

prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
msg = prompt.format(product="colorful socks")
print(msg)
# What is a good name for a company that makes colorful socks?


prompt_template = PromptTemplate.from_template(
    "Tell me a {adjective} joke about {content}."
)
print(prompt_template.format(adjective="funny", content="chickens"))
# Tell me a funny joke about chickens.


template = "You are a helpful assistant that translates {input_language} to {output_language}."
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])

msg2 = chat_prompt.format_messages(input_language="English", output_language="French", text="I love programming.")
print(msg2)
# [SystemMessage(content='You are a helpful assistant that translates English to French.'),
# HumanMessage(content='I love programming.')]


chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI bot. Your name is {name}."),
        ("human", "Hello, how are you doing?"),
        ("ai", "I'm doing well, thanks!"),
        ("human", "{user_input}"),
    ]
)

messages = chat_template.format_messages(name="Bob", user_input="What is your name?")
print(messages)
# [SystemMessage(content='You are a helpful AI bot. Your name is Bob.'),
# HumanMessage(content='Hello, how are you doing?'),
# AIMessage(content="I'm doing well, thanks!"),
# HumanMessage(content='What is your name?')]



chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You are a helpful assistant that re-writes the user's text to "
                "sound more upbeat."
            )
        ),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
)
messages = chat_template.format_messages(text="I don't like eating tasty things")
print(messages)
# [SystemMessage(content="You are a helpful assistant that re-writes the user's text to sound more upbeat."),
# HumanMessage(content="I don't like eating tasty things")]


prompt_val = prompt_template.invoke({"adjective": "funny", "content": "chickens"})
print(prompt_val) #text='Tell me a funny joke about chickens.'
print(StringPromptValue(text='Tell me a joke')) #text='Tell me a joke'
print(prompt_val.to_string()) #Tell me a funny joke about chickens.
print(prompt_val.to_messages()) #[HumanMessage(content='Tell me a funny joke about chickens.')]

chat_val = chat_template.invoke({"text": "i dont like eating tasty things."})
print(chat_val)
#messages=[SystemMessage(content="You are a helpful assistant that re-writes the user's text to sound more upbeat."),
# HumanMessage(content='i dont like eating tasty things.')]
print(chat_val.to_string())
#System: You are a helpful assistant that re-writes the user's text to sound more upbeat.
#Human: i dont like eating tasty things.



