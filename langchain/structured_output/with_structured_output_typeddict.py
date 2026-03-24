from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Annotated

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")

# schema
class Review(TypedDict):
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[str, "A sentiment of the review, either negative, positive or neutral"]

model_with_structured_output =  model.with_structured_output(Review)

result = model_with_structured_output.invoke("""Great location, beautiful surrounding atmosphere, and the most friendly and helpful receptionist. The room was clean, stylish, and comfortable. I was very pleased with my stay and hope to be back.""")

print(result)
