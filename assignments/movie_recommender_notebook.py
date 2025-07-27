
# Movie Recommender System using LangChain, LLM, and Vector Store

## 1. Setup & Installation

```python
!pip install langchain openai pandas chromadb sentence-transformers
```

## 2. Load and Explore Dataset

```python
import pandas as pd

# Load dataset
df = pd.read_csv("your_dataset.csv")

# Display basic info
df.info()
df.head()
```

## 3. Data Preprocessing

```python
# Fill missing values
df = df.fillna("")

# Combine relevant columns for vectorization
df["combined_text"] = df.apply(lambda row: f"{row['orig_title']} ({row['date_x']}): {row['genre']}. {row['overview']}. Crew: {row['crew']}. Country: {row['country']}", axis=1)

# Convert to list of documents
docs = df["combined_text"].tolist()
```

## 4. Vectorization

```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

# Use sentence-transformers for embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create Document objects
documents = [Document(page_content=text) for text in docs]

# Vector store setup
vectorstore = Chroma.from_documents(documents, embedding_model, persist_directory="./chroma_store")
```

## 5. Define LLM for Query Processing

```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

prompt_template = PromptTemplate(
    input_variables=["query"],
    template="""Extract relevant filters like genre, date range, actor or crew from this movie query:
Query: {query}
Respond as JSON with keys like genre, date_range, actor, crew.
""")

filter_chain = LLMChain(llm=llm, prompt=prompt_template)
```

## 6. Define Search Function

```python
def generate_search_query(query):
    filter_info = filter_chain.run(query)
    print("Extracted Filters:", filter_info)
    return query  # Optionally enhance query

def search_movies(query, top_k=5):
    user_query = generate_search_query(query)
    results = vectorstore.similarity_search(user_query, k=top_k)
    for i, res in enumerate(results):
        print(f"{i+1}. {res.page_content}\n{'-'*80}")
```

## 7. Example Usage

```python
search_movies("Find a comedy movie starring Jim Carrey from the 90s")
```

## 8. Optional: Enable Conversational Memory

```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

# Example follow-up
conversation.predict(input="Show me similar ones but more recent")
```
