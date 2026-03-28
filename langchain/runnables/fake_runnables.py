from abc import ABC, abstractmethod
import random

class Runnable(ABC):
    @abstractmethod
    def invoke(input):
        pass

class FakeLLM(Runnable):

    def __init__(self):
        pass

    def invoke(self, input):
        response_list = [
            "Delhi is the capital of india",
            "IPL is a cricket league",
            "AI stands for artificial intellegence"
        ]
        return {"response": {"choice": random.choice(response_list), "input": input}} 

llm = FakeLLM()

class FakePromptTemplate(Runnable):
    
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input):
        return self.template.format(**input)
    
template = FakePromptTemplate(
    template="Write about topic {topic}",
    input_variables=["topic"]
)

template_1 = FakePromptTemplate(
    template="Write about topic {choice}",
    input_variables=["choice"]
)

class FakeStrOutputParser(Runnable):

    def __init__(self):
        pass

    def invoke(self, input):
        return input["response"]

parser = FakeStrOutputParser()

class RunnableConnector(Runnable):

    def __init__(self, runnable_list):
        self.runnable_list = runnable_list

    def invoke(self, input):
        for runnable in self.runnable_list:
            input = runnable.invoke(input)
        return input


chain = RunnableConnector([template, llm, parser])

# result = chain.invoke({"topic": "HAjshajksdhjk"})

# print(result)
chain_1 = RunnableConnector([template_1, llm, parser])

connected_chains = RunnableConnector([chain, chain_1])

result = connected_chains.invoke({"topic": "Ai engineering"})

print(result)

