from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import os

os.environ['HF_HOME'] = "D:/huggingface_cache"


embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

documents = [
   "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers.",
    "Afzal lives in dayadara and he is a software engineer"
]

document_embeddings = embeddings.embed_documents(documents)

query = input("> ")

query_embeddings = embeddings.embed_query(query)

scores = cosine_similarity([query_embeddings], document_embeddings)[0]

index,  score = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]

# print(f"scores: {str(scores)}")
# print(f"score: {score}")
print(documents[index])

# query = input("> ")

# query_embeddings = embeddings.embed_query(query)

