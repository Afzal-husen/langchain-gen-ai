from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
# import os

load_dotenv()

# os.environ["HF_HOME"] = "D:/huggingface_cache"

# llm = HuggingFacePipeline.from_model_id(
#     model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation"
# )

# model = ChatHuggingFace(llm=llm)

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


# prompt_1 =  prompt_template_1.invoke({"topic": "AI engineering"})

# message_1 = model.invoke(prompt_1)

# result_1 = parser.invoke(message_1)

# prompt_2 =  prompt_template_2.invoke({"report": result_1})

# message_2 = model.invoke(prompt_2)

# result_2 = parser.invoke(message_2)

# chains- alternative of the above 
chain = prompt_template_1 | model | parser | prompt_template_2 | model | parser

result = chain.invoke({"topic": "AI engineering"})

print(result)





