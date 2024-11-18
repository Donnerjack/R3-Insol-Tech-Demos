from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')

index_name = "r3-demo"
file_path = r"D:\VSC Projects\R3-Insol-Tech-Demos\judgment.pdf"

# Load document
loader = PyPDFLoader(file_path)
text_documents = loader.load()

# Add file name to metadata
for doc in text_documents:
    doc.metadata["source"] = os.path.basename(file_path)

# Initialize text splitter and create chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True
)

chunks = text_splitter.split_documents(text_documents)

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=openai_api_key
)

# Create vector store
vectordb = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=index_name
)