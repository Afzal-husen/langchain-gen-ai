from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableBranch, RunnableSequence, RunnableLambda, RunnableParallel, RunnablePassthrough

load_dotenv()

def word_counter(text: str):
    return len(text.split())

model = ChatGroq(model="openai/gpt-oss-20b")

parser = StrOutputParser()

prompt_1 = PromptTemplate(
    template="Generate a detailed report on topic {topic}",
    input_variables=["topic"]
)

prompt_2 = PromptTemplate(
    template="Summarize the report {report}",
    input_variables=["report"]
)


report_gen_chain = RunnableSequence(prompt_1, model, parser)

summary_gen_chain = RunnableSequence(prompt_2, model, parser)

branch_chain = RunnableBranch(
    (lambda x: word_counter(x) > 500, summary_gen_chain),
   RunnablePassthrough()
)

chain = RunnableSequence(report_gen_chain, branch_chain)

result = chain.invoke({"topic": "AI"})

print(result)
