from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda


load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")

str_parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the feedback")

pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)

structured_output_model = model.with_structured_output(Feedback) 

prompt_1 = PromptTemplate(
    template="Generate a sentiment for the following feedback \n {feedback}",
    input_variables=["feedback"],
)

classifier_chain = prompt_1 | structured_output_model 


prompt_2 = PromptTemplate(
    template="Write an appropriate response for this positive feedback \n {feedback}",
    input_variables=["feedback"]
)
prompt_3 = PromptTemplate(
    template="Write an appropriate response for this negative feedback \n {feedback}",
    input_variables=["feedback"]
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == "positive", prompt_2 | model | str_parser),
    (lambda x:x.sentiment == "negative", prompt_3 | model | str_parser),
    RunnableLambda(lambda x:"No sentiment found")
)

chain = classifier_chain | branch_chain

result = chain.invoke({"feedback": "This is a good phone"})

print(result)






