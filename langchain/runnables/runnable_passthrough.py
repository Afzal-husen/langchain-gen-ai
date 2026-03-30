from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableParallel

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

joke_gen_chain = RunnableSequence(prompt_1, model, parser)
joke_explanation_chain = RunnableSequence(prompt_2, model, parser)

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "explanation": joke_explanation_chain
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({"topic", "Rats"})

print(result)