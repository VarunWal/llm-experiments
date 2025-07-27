
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

# Create metadata for filtering
from langchain.schema import Document
documents = []
for i, row in df.iterrows():
    metadata = {
        "genre": row["genre"],
        "crew": row["crew"],
        "orig_lang": row["orig_lang"],
        "country": row["country"],
        "date": row["date_x"]
    }
    documents.append(Document(page_content=row["combined_text"], metadata=metadata))
```

## 4. Vectorization

```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

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
    template="""Extract structured filters from this movie query.
Return a JSON with keys like genre, date_range, actor (crew), orig_lang, country.

Query: {query}
""")

filter_chain = LLMChain(llm=llm, prompt=prompt_template)
```

## 6. Define Search with RAG Filters

```python
def parse_filter_string(filter_json):
    try:
        import json
        filters = json.loads(filter_json)
        return filters
    except Exception as e:
        print("Error parsing filters:", e)
        return {}

def search_movies_with_filters(query, top_k=5):
    filter_response = filter_chain.run(query)
    filters = parse_filter_string(filter_response)
    print("Applied Filters:", filters)

    # Build filter function
    def metadata_filter(metadata):
        checks = []
        if "genre" in filters:
            checks.append(filters["genre"].lower() in metadata.get("genre", "").lower())
        if "actor" in filters:
            checks.append(filters["actor"].lower() in metadata.get("crew", "").lower())
        if "country" in filters:
            checks.append(filters["country"].lower() in metadata.get("country", "").lower())
        if "orig_lang" in filters:
            checks.append(filters["orig_lang"].lower() in metadata.get("orig_lang", "").lower())
        if "date_range" in filters:
            try:
                year = int(str(metadata.get("date", "0"))[:4])
                checks.append(filters["date_range"][0] <= year <= filters["date_range"][1])
            except:
                pass
        return all(checks)

    results = vectorstore.similarity_search_with_score(query, k=top_k)
    filtered_results = [(doc, score) for doc, score in results if metadata_filter(doc.metadata)]

    for i, (doc, score) in enumerate(filtered_results):
        print(f"{i+1}. {doc.page_content}\nScore: {score}\n{'-'*80}")
```

## 7. Example Usage

```python
search_movies_with_filters("Find a French comedy movie from the early 2000s with Gérard Depardieu")
```
