from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")

chat_history = []

while True:
    user_input = input("You: ")
    
    if user_input.lower() == "exit":
        break

    chat_history.append({"role": "user", "content": user_input})
    
    result = model.invoke(chat_history)

    chat_history.append({"role": "ai", "content": chat_history})
    
    print("AI: ", result.content)

print(chat_history)

