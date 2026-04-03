from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

code = """import random

class FakeLLM():
    def __init__(self):
        print("LLM created")

    def predict(self, prompt):

        response_list = [
            "Delhi is the capital of india",
            "IPL is a cricket league",
            "AI stands for artificial intellegence"
        ]

        return { "response":  {"llm_reponse": random.choice(response_list), "user_prompt": prompt}  }

llm = FakeLLM()


# print(result)

class FakePromptTemplate():

    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables
    
    def format_template(self, input_dict):
        return self.template.format(**input_dict)

template = FakePromptTemplate(
    template="Write about the topic {topic}",
    input_variables=["topic"]
)


        
class FakeChain():

    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt

    def run(self, input_dict): 
        final_prompt = self.prompt.format_template(input_dict)
        result = llm.predict(final_prompt)
        return result["response"]

chain = FakeChain(llm, template)

result = chain.run({"topic": "Africa"})

print(result)
        
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=500,
    chunk_overlap=0
)

chunks = splitter.split_text(code)

print(chunks[0])