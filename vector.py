from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain


print("1")

with open("vk3.txt", "r",  encoding="utf-8") as f:
    state_of_the_union = f.read()
print("3")

print("4")
embeddings = OllamaEmbeddings()

print("5")
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex= "\n",
)

print("6")

texts = text_splitter.create_documents([state_of_the_union])
documents = text_splitter.split_documents(texts)
print("7")

vector = FAISS.from_documents(documents, embeddings)
vector.save_local("vector_db2")
print("8")



