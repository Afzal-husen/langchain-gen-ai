from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

new_person:Person = {'name': "afzal", "age": 29}

print(new_person)