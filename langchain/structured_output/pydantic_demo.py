from pydantic import BaseModel, EmailStr, Field
from typing import  Optional

class Student(BaseModel):
    name: str = "afzal diwan"
    age: Optional[int] = None
    email: Optional[EmailStr] = None
    cgpa: float = Field(gt=0, lt=10, default=6)

s1 = {'name': "afzal", "age": 20, "email": "abc@gmail.com", "cgpa": 9.6}
s2 = {}

student1 = Student(**s1)
student2 = Student(**s2)

print(student1)
print(student2)