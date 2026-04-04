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

vector_store = Chroma(
    embedding_function=embedding,
    persist_directory="./vector_stores/my_db2",
    collection_name="copy-sample"
)

doc_ids = vector_store.add_documents(docs)

print(f"doc_ids: {doc_ids}")

while True:
    query = input("query: ")

    if query == "exit" or query == "quit":
        break

    result = vector_store.similarity_search(
        query=query,
        k=2
    )

    print(result[0].page_content)


