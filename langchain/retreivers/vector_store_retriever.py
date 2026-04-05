from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
import os

os.environ["HF_HOME"] = "D:/huggingface_cache"

embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

doc1 = Document(
    page_content="Afzal Diwan is a software engineer and he lives in Gujarat which is a state of India.",
    metadata={"location": "Gujarat, Indian"}
)

doc2 = Document(
    page_content="John smith is a senior software engineer working in google and he lives in silicon valley, sanfrancisco, USA.",
    metadata={"location": "Sanfrancisco, USA"}
)

docs = [doc1, doc2]

vector_store = Chroma.from_documents(
    embedding=embedding,
    collection_name='my_collection',
    documents=docs
)

retriever = vector_store.as_retriever(search_kwargs={"k": 1})

result = retriever.invoke("Who is afzal ?")

for i, doc in enumerate(result):
    print(f"Result - {i + 1}")
    print(f"Content: \n {doc.page_content}")

