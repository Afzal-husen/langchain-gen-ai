from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnableParallel, RunnablePassthrough

load_dotenv()

def word_counter(text: str):
    return len(text.split())

model = ChatGroq(model="openai/gpt-oss-20b")

prompt_1 = PromptTemplate(
    template="Generate a joke on the topic {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt_1, model, parser)

word_counter_runnable = RunnableLambda(word_counter)

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "word_count": word_counter_runnable
})

chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = chain.invoke({"topic": "computers"})

# final_result = """{} \n word count - {}""".format(result['joke'], result['word_count'])

final_result = """The joke: \n {} \n Word count: {}""".format(result["joke"], result["word_count"])
print(final_result)



