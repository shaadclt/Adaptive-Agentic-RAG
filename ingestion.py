from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from model import embed_model

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

doc_splits = text_splitter.split_documents(docs_list)

embed = embed_model

# Create vector store with documents
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embed,
    persist_directory="./.chroma")


# Create retriever
retriever = vectorstore.as_retriever()


"""
This ingestion pipeline forms the backbone of our local knowledge base. 
We start by loading environment variables, then define a curated list of URLs containing high-quality content about AI agents, prompt engineering, and adversarial attacks.
The WebBaseLoader fetches content from these URLs and loads them into document objects. 
We then use the RecursiveCharacterTextSplitter to break down these documents into smaller, manageable chunks of 250 tokens each, which is optimal for embedding and retrieval. 
The splitter uses tiktoken encoding to ensure accurate token counting.
Finally, we create a Chroma vector store that will persist our embeddings locally, using Google's text-embedding-004 model for high-quality semantic representations.
"""