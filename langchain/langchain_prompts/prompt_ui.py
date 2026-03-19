from langchain_groq import ChatGroq 
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")


st.header('Research tool')

user_input = st.text_input("Enter your prompt")

result =  model.invoke(user_input)

if st.button("Summarize"):
    st.write(result.content)