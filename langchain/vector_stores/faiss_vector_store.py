from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import os
import faiss

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

index = faiss.IndexFlatL2(len(embedding.embed_query("test")))

vector_store = FAISS(
    embedding_function=embedding,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

docs_ids = vector_store.add_documents(docs)

print(docs_ids)

result = vector_store.similarity_search(
    query="who is afzal ?",
    k=2
)

print(result[0].page_content)