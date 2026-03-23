from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b", temperature=2)

chat_prompt_template = ChatPromptTemplate([
    ('system', "You are an expert in {domain}"),
    ('user', "Explain in simple terms, what is  {topic}")
])


domain = input("What domain would you like to explore ? ")
topic = input(f"What topic would you like to explore in {domain} ? ")

prompt = chat_prompt_template.invoke({"domain": domain, "topic": topic})

result = model.invoke(prompt)
print(result.content)
    

 