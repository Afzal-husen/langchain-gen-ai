from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal, Optional

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")

class Review(BaseModel):
    key_themes: list[str] = Field(description="Write the list of key themes")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description= "A sentiment of the review, either negative, positive or neutral") 
    pros: Optional[list[str]] =  Field(default=None, description="Optional pros")
    cons: Optional[list[str]] =  Field(default=None, description="Optional cons")
    name: str =  Field(description="Name of the reviewer")

model_with_structured_output = model.with_structured_output(Review)

result = model_with_structured_output.invoke("""Great location, beautiful surrounding atmosphere, and the most friendly and helpful receptionist. The room was clean, stylish, and comfortable. I was very pleased with my stay and hope to be back.
Review by Afzal
""")

print(result)
