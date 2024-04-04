from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain 
from langchain import hub



print("1")
embeddings = OllamaEmbeddings()
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

print("2")

llm = Ollama(model="llama2")

document_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
print("9")

print("10")

retriever = FAISS.load_local("vector_db", embeddings, allow_dangerous_deserialization=True).as_retriever()
print("11")

retrieval_chain = create_retrieval_chain(retriever, document_chain)
print("12")

response = retrieval_chain.invoke({"input": "who is virat kohli?"})
print("13")

print(response["answer"])
print("14")