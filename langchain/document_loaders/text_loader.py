from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Generate the summary for the poem \n {poem}",
    input_variables=["poem"]
)

chain = prompt | model | parser

loader = TextLoader("./document_loaders/cricket.txt", encoding="utf-8")

docs = loader.load()

result =chain.invoke({"poem": docs[0].page_content})

print(result)