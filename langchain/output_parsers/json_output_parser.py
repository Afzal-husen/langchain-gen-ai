from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")

parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me name, age, summary and city of a fictional character {character} \n {format_instruction}',
    input_variables=["character"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)


# prompt = template.format(character="sung jinwoo from solo leveling")

# result = model.invoke(prompt)

# print(result.content)

# ALTERNATIVE
chain = template | model | parser

final_output = chain.invoke({"character":"sung jinwoo from solo leveling"})

print(final_output)
print(type(final_output))