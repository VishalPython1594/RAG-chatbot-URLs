import os
import streamlit as st
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from langchain_community.document_loaders import WebBaseLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from bs4 import BeautifulSoup
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

# Load API Key Safely
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
USER_AGENT = os.getenv("USER_AGENT")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Check your .env file.")

if not USER_AGENT:
    raise ValueError("USER_AGENT not found. Check your .env file.")

# Initialize NLTK resources
nltk.download('stopwords')
os.system("python -m nltk.downloader punkt")
nltk.download('punkt')
nltk.download('wordnet')

# Text Cleaning Function
lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

def cleaned_data(docs):
    doc_clean = re.sub(r'http\S+|www\S+', '', docs)  # Remove URLs
    doc_clean = re.sub(r'[^\w\s]', '', doc_clean)  # Remove punctuation but keep numbers
    doc_token = nltk.word_tokenize(doc_clean)
    docs_cleaned = [lemmatizer.lemmatize(doc.lower()) for doc in doc_token if doc.lower() not in stop_words]
    return ' '.join(docs_cleaned)

# Streamlit UI
st.title("Web Content Q&A Tool")

urls = st.text_area("Enter URLs (comma-separated)", "")

# Initialize session state for storing loaded data
if "loaded_data" not in st.session_state:
    st.session_state.loaded_data = None  # Store fetched content
if "retriever" not in st.session_state:
    st.session_state.retriever = None   # Store retriever
if "chain" not in st.session_state:
    st.session_state.chain = None       # Store the chain

if st.button("Load Content"):
    urls = [url.strip() for url in urls.split(',') if url.strip()]
    try:
        loader = WebBaseLoader(urls, header_template={"User-Agent": USER_AGENT})
        data = loader.load()
        
        if not data:
            st.error("No data was fetched. Check the URLs.")
        else:
            st.session_state.loaded_data = data  # Store in session state
            st.success("Data Loaded Successfully!")
    except Exception as e:
        st.error(f"Error fetching data: {e}")

# Ensure `loaded_data` exists before processing
if st.session_state.loaded_data:
    data = st.session_state.loaded_data
    data_content = [cleaned_data(doc.page_content) for doc in data]
    documents = [Document(page_content=content, metadata=doc.metadata) for content, doc in zip(data_content, data)]

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=95)
    chunks = text_splitter.split_documents(documents)

    # Embedding & Vector Store Initialization
    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Ensure persistent directory exists
    PERSIST_DIR = 'chroma_db'
    if not os.path.exists(PERSIST_DIR):
        os.makedirs(PERSIST_DIR)

    # Load or create database
    db = Chroma(
        collection_name='vect-db',
        embedding_function=hf_embeddings,
        persist_directory=PERSIST_DIR
    )

    if chunks:
        db.add_documents(chunks)
        
        # Create retriever and store in session state
        st.session_state.retriever = db.as_retriever(search_type='similarity', search_kwargs={'k': 3})
        
        prompt_template = '''
        You are an AI assistant that answers questions only based on the content from the provided websites. 
        Do not use any external knowledge. If the answer is not present in the provided data, simply say "I don't know."
        {context}
        Answer the question: {question}
        '''

        chat_model = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model='gemini-1.5-flash')
        prompt_temp = ChatPromptTemplate.from_template(prompt_template)
        output_parser = StrOutputParser()

        chain = {'context': st.session_state.retriever | (lambda docs: '\n\n'.join(doc.page_content for doc in docs)), 
                'question': RunnablePassthrough()} | prompt_temp | chat_model | output_parser

        st.session_state.chain = chain  # Store the chain in session state

    else:
        st.error("No valid document chunks to process.")

# Question Input
query = st.text_input("Ask a Question")

if st.button("Get Answer"):
    if query and st.session_state.chain:
        answer = st.session_state.chain.invoke(query)
        st.write("### Answer:")
        st.write(answer)
    else:
        st.error("Please load content first before asking a question.")

