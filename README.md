# Web Content Q&A Tool
In my Web Content Q&A Tool, I follow a structured process to ingest content from web pages, store it efficiently, and retrieve relevant information to generate answers. Here’s how it works:

## Step 1: Ingesting Web Content

### 1. User Input:

* The user enters one or more URLs in the input field.
  
### 2.Fetching Content:

* I use LangChain’s WebBaseLoader to scrape and extract content from the provided URLs.
* The user-agent is passed to avoid request blocking.
```python
loader = WebBaseLoader(urls, header_template={"User-Agent": USER_AGENT})
data = loader.load()
```

### 3.Text Cleaning:

* The raw content is processed to remove:
  * URLs
  * Punctuation
  * Stopwords
  * Unnecessary whitespace
* Tokenization and lemmatization are applied to standardize words.
```python
def cleaned_data(docs):
    doc_clean = re.sub(r'http\S+|www\S+', '', docs)
    doc_clean = re.sub(r'[^\w\s]', '', doc_clean)
    doc_token = nltk.word_tokenize(doc_clean)
    docs_cleaned = [lemmatizer.lemmatize(doc.lower()) for doc in doc_token if doc.lower() not in stop_words]
    return ' '.join(docs_cleaned)
```

## Step 2: Storing Processed Content in a Vector Database

### 1. Chunking the Data:

* Large documents are split into smaller, overlapping chunks using RecursiveCharacterTextSplitter.
* This improves retrieval accuracy.
```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=95)
chunks = text_splitter.split_documents(documents)
```

### 2.Embedding the Chunks:

* I use Hugging Face’s all-MiniLM-L6-v2 model to convert text into numerical vectors.
* These embeddings help in retrieving relevant content efficiently.
```python
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

### 3.Storing in ChromaDB:

* I used ChromaDB, a persistent vector database, to store and index the embeddings.
* This ensures fast and efficient retrieval.
```python
db = Chroma(
    collection_name='vect-db',
    embedding_function=hf_embeddings,
    persist_directory=PERSIST_DIR
)
db.add_documents(chunks)
```

## Step 3: Retrieving Content & Generating Answers

### 1. Retrieval of Relevant Chunks:

* When a user asks a question, I use vector similarity search to find the most relevant chunks from the stored content.
```python
st.session_state.retriever = db.as_retriever(search_type='similarity', search_kwargs={'k': 3})
```

### 2. Prompt Engineering & AI Model Response:

* A structured prompt is created to instruct the AI to answer only based on retrieved content (no external knowledge).
* I use Google Gemini 1.5 Flash for generating responses.
```python
prompt_template = '''
You are an AI assistant that answers questions only based on the content from the provided websites. 
Do not use any external knowledge. If the answer is not present in the provided data, simply say "I don't know."
{context}
Answer the question: {question}
'''

chat_model = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model='gemini-1.5-flash')
```

### 3. Generating Answers:

* The retrieved chunks are passed to the AI model via a structured processing pipeline.
* The AI generates a concise, accurate response.
```python
chain = {'context': st.session_state.retriever | (lambda docs: '\n\n'.join(doc.page_content for doc in docs)), 
         'question': RunnablePassthrough()} | prompt_temp | chat_model | output_parser
answer = chain.invoke(query)
```

## Final Outcome
* The user enters URLs, and I ingest and clean the content.
* I stored the processed content as vector embeddings in a database.
* When a question is asked, I retrieve relevant information and generate a precise answer using Google Gemini.
  
This ensures that answers are always based on the ingested content and not on general knowledge.


## Instructions to Run the App Locally
Follow these steps to set up and run the Web Content Q&A Tool on your local machine.

### 1. Clone the Repository (If Applicable)
If your code is stored in a repository (e.g., GitHub), clone it using:
```bash
git clone https://github.com/VishalPython1594/RAG-chatbot-URLs.git
cd RAG-chatbot-URLs
```
### 2. Set Up a Virtual Environment (Recommended)
It's best practice to use a virtual environment to avoid dependency conflicts.

For Windows (Command Prompt or PowerShell):

```bash
python -m venv venv
venv\Scripts\activate
```

For Mac/Linux (Terminal):

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Ensure all required packages are installed by running:

```bash
pip install -r requirements.txt
```
If you don’t have a requirements.txt file, install the necessary packages manually:

```bash
pip install streamlit nltk beautifulsoup4 langchain langchain-community langchain-google-genai langchain-chroma chromadb sentence-transformers python-dotenv
```

### 4. Set Up Environment Variables
Create a .env file in the project directory and add the following:

```ini
GOOGLE_API_KEY=your_google_api_key_here
USER_AGENT=your_custom_user_agent_string
```
#### Note:- You can use "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" as your USER_AGENT

Replace your_google_api_key_here with your actual API key and your_custom_user_agent_string with a valid user-agent string.

### 5. Download Required NLTK Data
Run the following commands to ensure necessary NLTK components are available:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

### 6. Run the Streamlit App
Start the Streamlit app using:

```bash
streamlit run app.py
```

### 7. Access the Web Interface
Once the app starts, you will see a URL in the terminal similar to:

```nginx
Local URL: http://localhost:8501
```

Open this URL in your web browser to interact with the application.

### 8. Troubleshooting

* If you encounter missing dependencies, re-run pip install -r requirements.txt.
* Ensure the .env file is correctly set up with valid API keys.
* If the app crashes, check for error messages and verify all installations.


## 🤝 Contributing
Contributions are welcome! Feel free to submit a pull request if you have improvements or new features.

## 📜 License
This project is licensed under the MIT License.

## 📩 Contact & Support
📧 Email: vishal1594@outlook.com 🔗 LinkedIn: https://www.linkedin.com/in/vishal-shivnani-87487110a/
