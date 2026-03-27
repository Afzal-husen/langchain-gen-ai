from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")

parser = StrOutputParser()

template_1 = PromptTemplate(
    template="Give detailed report on topic {topic}",
    input_variables=["topic"]
)

template_2 = PromptTemplate(
    template="Extract 5 most important points from report {report}",
    input_variables=["report"]
)

chain = template_1 | model | parser | template_2 | model | parser

result = chain.invoke({"topic": "How AI is affecting Software engineering market"})

print(result)
print(chain.get_graph().print_ascii())


