from langchain_community.document_loaders import CSVLoader

loader = CSVLoader("./document_loaders/Social_Network_Ads.csv")

docs = loader.load()

print(docs[0].page_content)