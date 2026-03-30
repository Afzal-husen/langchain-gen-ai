from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")

prompt_1 = PromptTemplate(
    template="Write a joke on a topic {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()



# # Joke explanation

prompt_2 = PromptTemplate(
    template="Write the explanation for the joke {joke}",
    input_variables=["joke"]
)

chain = RunnableSequence(prompt_1, model, parser)
chain_2 = RunnableSequence(prompt_2, model, parser)

final_chain = RunnableSequence(chain, chain_2)

result = final_chain.invoke({"topic", "AI"})

print(result)