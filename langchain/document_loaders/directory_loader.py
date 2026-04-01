from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path="./document_loaders/books",
    glob="*.pdf", # **/*.pdf, **/*, "data/*.csv"
    loader_cls=PyPDFLoader
)

# docs =  loader.load() #eager loading (loads all the documents at once)
# print(docs[1].page_content)

docs = loader.lazy_load()  # lazy loader (loads documents one by one per page) user for large files

for document in docs:
    print(document.metadata)
