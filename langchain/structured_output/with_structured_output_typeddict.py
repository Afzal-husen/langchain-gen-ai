from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal, Optional

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")

# schema
class Review(TypedDict):
    key_themes: Annotated[list[str], "Write the list of key themes"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["pos", "neg"], "A sentiment of the review, either negative, positive or neutral"]
    pros: Annotated[Optional[str], "Optional pros"]
    cons: Annotated[Optional[str], "Optional cons"]
    name: Annotated[str, "Name of the reviewer"]

model_with_structured_output =  model.with_structured_output(Review)

result = model_with_structured_output.invoke("""Great location, beautiful surrounding atmosphere, and the most friendly and helpful receptionist. The room was clean, stylish, and comfortable. I was very pleased with my stay and hope to be back.
Review by Afzal
""")

print(result)
