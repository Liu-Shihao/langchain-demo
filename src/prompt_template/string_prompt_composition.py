from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


prompt = (
    PromptTemplate.from_template("Tell me a joke about {topic}")
    + ", make it funny"
    + "\n\nand in {language}"
)

prompt.format(topic="sports", language="spanish")

model = ChatOpenAI()


chain = LLMChain(llm=model, prompt=prompt)