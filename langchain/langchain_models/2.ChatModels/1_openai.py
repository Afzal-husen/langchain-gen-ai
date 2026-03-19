from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

"""
params
model, temperature (0-2), max_completion_tokens
"""
llm = ChatOpenAI(model="gpt-3.5-turbo-instruct", temperature=0, max_completion_tokens=10)

result = llm.invoke("what is the capital of india ?")

print(result)