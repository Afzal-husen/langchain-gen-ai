from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")

parser = StrOutputParser()

prompt_template_1 = PromptTemplate(
    template="Give a detailed report on the topic {topic}",
    input_variables=["topic"],
    validate_template=True
)

prompt_template_2 = PromptTemplate(
    template="Write 5 line summary based on the report {report}",
    input_variables=["report"],
    validate_template=True
)

prompt_1 =  prompt_template_1.invoke({"topic": "AI engineering"})

message_1 = model.invoke(prompt_1)

result_1 = parser.invoke(message_1)

prompt_2 =  prompt_template_2.invoke({"report": result_1})

message_2 = model.invoke(prompt_2)

result_2 = parser.invoke(message_2)

print(result_2)




