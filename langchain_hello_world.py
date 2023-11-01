from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
chat_model = ChatOpenAI()

print(chat_model.predict('Hi'));

system_template = """You are a helpful assistant who returns comma seperated lists. You take in a category and return a csv with five objects. ONLY return csv and nothing else."""

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template = "{text}"

human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])

# Parser
class CSVParser(BaseOutputParser):
    def parse(self,text:str):
        print("In csv parser",text)
        return text.strip().split(", ")

chain = LLMChain(
    llm = ChatOpenAI(),
    prompt = chat_prompt,
    output_parser=CSVParser()
)

print(chain.run("Colours"));