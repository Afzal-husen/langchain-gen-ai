from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate

chat_prompt_template = ChatPromptTemplate([
    ('system', 'You are a helpfull ai assitant'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('user', '{query}')
])

chat_history = []

with open("./langchain_prompts/chat_history.txt") as f:
    chat_history.extend(f.readlines())

print(chat_history)

prompt = chat_prompt_template.invoke({"chat_history": chat_history, "query": "where is my refund ?"})

print(prompt)

