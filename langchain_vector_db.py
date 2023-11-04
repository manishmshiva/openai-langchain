from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Tea")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_documents(texts,embeddings)

QA = RetrievalQA.from_chain_type(
    llm=OpenAI(),chain_type="stuff",retriever = docsearch.as_retriever())

while True:
    query = input("Ask a question about Tea.\n")
    if(query == 'exit'):
        break
    else:
        print(QA.run(query))