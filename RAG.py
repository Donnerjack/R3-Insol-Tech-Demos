import streamlit as st
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA

#Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')

#Streamlit configuration
st.set_page_config(page_title="RAG", page_icon="📚")
st.title("RAG on Judgments.")

#Initialise components
@st.cache_resource
def initialize_qa_chain():
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = PineconeVectorStore(
        index_name="r3-demo",
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever()
    
    model = ChatOpenAI(
        model="gpt-4",
        temperature=0
    )
    
    return RetrievalQA.from_chain_type(
        model,
        retriever=retriever,
        return_source_documents=True
    )

qa_chain = initialize_qa_chain()

# Query function
def query_document(question: str):
    with st.spinner('Searching for answer...'):
        result = qa_chain({"query": question})
        return result['result'], [doc.metadata['source'] for doc in result['source_documents']]

# User interface
question = st.text_input("Ask question:")

if question:
    answer, sources = query_document(question)
    
    st.markdown("### Answer")
    st.write(answer)
    
    st.markdown("### Sources")
    for source in set(sources):
        st.info(f"📄 {source}")

# Footer
st.markdown("---")
st.markdown("*DQKC & ADV - using LangChain, OpenAI, Streamlit and Pinecone")