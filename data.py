import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


file = open("combined.txt", "r")
contents = file.read()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap  = 20,
    length_function = len,
)
texts = text_splitter.create_documents([contents])

print("Beginning construction of FAISS DB")
docs = FAISS.from_documents(texts, embeddings)

print("Beginning pickle")
with open("docs.pkl", "wb") as f:
    pickle.dump(docs, f)
print("pickle over")