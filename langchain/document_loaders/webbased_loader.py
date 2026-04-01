from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Answer the question {question} for the following \n {content}",
    input_variables=["question", "content"]
)

chain = prompt | model | parser

url = "https://reference.langchain.com/python/langchain/overview"

loader = WebBaseLoader(url)

docs = loader.load()

# print(docs[0].page_content)

result = chain.invoke({"question": "What is the content about ?", "content": docs[0].page_content})

print(result)