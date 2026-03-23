from langchain.messages import SystemMessage, AIMessage, HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=2
    )

chat_history = [
    SystemMessage(content="You are a helpfull assitant")
]

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit": 
        break

    chat_history.append(HumanMessage(content=user_input))

    result =  model.invoke(chat_history)

    chat_history.append(AIMessage(content=result.content))
    print("AI: ", result.content)

print(chat_history)

