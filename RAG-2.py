import streamlit as st
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')

# Streamlit configuration
st.set_page_config(page_title="RAG", page_icon="📚")
st.title("RAG on Judgments")

# Initialize components
@st.cache_resource
def initialize_qa_chain(k_documents=4):  # Add k_documents parameter
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = PineconeVectorStore(
        index_name="r3-demo",
        embedding=embeddings
    )
    # Configure retriever with specific k
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_documents})
    
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0
    )
    
    return RetrievalQA.from_chain_type(
        model,
        retriever=retriever,
        return_source_documents=True
    )

# Add a number input for k_documents
k_documents = st.sidebar.number_input(
    "Number of chunks to retrieve",
    min_value=1,
    max_value=20,
    value=4,
    help="Select how many relevant document chunks to retrieve for each query"
)

qa_chain = initialize_qa_chain(k_documents)

# Modified query function to show chunks
def query_document(question: str):
    with st.spinner('Searching for answer...'):
        result = qa_chain({"query": question})
        return (
            result['result'],
            [(doc.metadata['source'], doc.page_content) for doc in result['source_documents']]
        )

# User interface
question = st.text_input("Ask question:")

if question:
    answer, sources_and_chunks = query_document(question)
    
    st.markdown("### Answer")
    st.write(answer)
    
    st.markdown(f"### Sources (Retrieved {len(sources_and_chunks)} chunks)")
    for i, (source, chunk) in enumerate(sources_and_chunks, 1):
        with st.expander(f"📄 Chunk {i} from {source}"):
            st.text(chunk)

# Footer
st.markdown("---")
st.markdown("*DQKC & ADV - using LangChain, OpenAI, Streamlit and Pinecone")