from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./text-splitters/dl-curriculum.pdf")

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=5,
    separator=""
)

docs = loader.load()

# text = """An intuitive strategy is to split documents based on their length. This simple yet effective approach ensures that each chunk doesn’t exceed a specified size limit. Key benefits of length-based splitting:
# Straightforward implementation
# Consistent chunk sizes
# Easily adaptable to different model requirements
# Types of length-based splitting:
# Token-based: Splits text based on the number of tokens, which is useful when working with language models.
# Character-based: Splits text based on the number of characters, which can be more consistent across different types of text.
# """

# result = splitter.split_text(text)

result = splitter.split_documents(docs)

print(result[0].page_content)