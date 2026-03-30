from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnableSequence

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")

prompt_1 = PromptTemplate(
    template="Generate a tweet to for tweeter on the topic {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()



# # Joke explanation

prompt_2 = PromptTemplate(
    template="Generate a content for a linkedn post on the topic {topic}",
    input_variables=["topic"]
)

chain = RunnableSequence(prompt_1, model, parser)
chain_2 = RunnableSequence(prompt_2, model, parser)

parallel_chain = RunnableParallel({
    "tweet": chain,
    "linkedn": chain_2,
})

result = parallel_chain.invoke({"topic", "AI"})

print(result)