from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda

load_dotenv()

model = ChatGroq(model="qwen/qwen3-32b")

str_parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the feedback")
    feedback: str = Field(description="Feedback provided by user")

pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)

prompt_1 = PromptTemplate(
    template="Generate a sentiment for the following feedback {feedback} \n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction": pydantic_parser.get_format_instructions()}
)

classifier_chain = prompt_1 | model | pydantic_parser


prompt_2 = PromptTemplate(
    template="Write an appropriate response for this {sentiment} feedback {feedback} \n {format_instruction}",
    input_variables=["sentiment", "feedback"],
    partial_variables={"format_instruction": pydantic_parser.get_format_instructions()}
)

# prompt_3 = PromptTemplate(
#     template="Write a negative response for the feedback \n {feedback}",
#     input_variables=["feedback"]
# )

pos_chain = prompt_2 | model | str_parser

# neg_chain = prompt_3 | model | str_parser

# chain = classifier_chain | pos_chain

# print(pos_chain.invoke({"feedback": "This is a really good phone!", "sentiment": "positive"}))

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == "positive", pos_chain),
    (lambda x:x.sentiment == "negative", pos_chain),
    RunnableLambda(lambda x: "Could not find an appropriate response")
)


chain = classifier_chain | branch_chain

print(chain.invoke({"feedback": "This is a really good phone!"}))

# result = chain.invoke({"feedback": "This is a really good phone!"})

# print(result)