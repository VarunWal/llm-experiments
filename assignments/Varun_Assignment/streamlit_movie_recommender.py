import os
import json
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_mistralai import ChatMistralAI
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
from typing import Optional, List

# Pydantic model for filter validation
class MovieFilters(BaseModel):
    genre: Optional[str] = Field(None, description="Genre of the movie")
    date_range: Optional[str] = Field(None, description="Release date range for the movie")
    actors: Optional[List[str]] = Field(None, description="List of actors in the movie")
    plot_details: Optional[list] = Field(None, description="Specific plot elements or keywords")

# Set up Streamlit UI
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Natural Language Movie Recommender")

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'last_results' not in st.session_state:
    st.session_state.last_results = []
if 'conversation_memory' not in st.session_state:
    st.session_state.conversation_memory = ConversationBufferMemory()

# Sidebar config
query = st.text_input("Enter your movie query or followup question:", 
                     placeholder="e.g., A sci-fi movie with time travel from the 90s")

# API setup
api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-small-latest"

# Initialize Mistral LLM
llm = ChatMistralAI(model_name=model, temperature=0.3)

# Enhanced prompt template for extracting filters
filter_prompt = PromptTemplate(
    input_variables=["query", "conversation_history"],
    template="""Based on the user's query and conversation history, extract relevant movie filters. 
Return ONLY a valid JSON object with the following possible keys: genre, date_range, actors, plot_details.
Do NOT include any additional text, explanations, or markdown formatting like ```json```.

Previous conversation:
{conversation_history}

Current query: {query}

Extracted filters as JSON:
"""
)

filter_chain = LLMChain(llm=llm, prompt=filter_prompt)

# Load vector store
@st.cache_resource
def load_vector_store():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./chroma_store", embedding_function=embedding_model)
    return vectorstore

vectorstore = load_vector_store()

def parse_filters(json_str: str) -> MovieFilters:
    """Parse and validate filters using Pydantic model"""
    try:
        # First try to parse the JSON string
        json_data = json.loads(json_str)
        
        # Then validate against our Pydantic model
        filters = MovieFilters(**json_data)
        return filters
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON format: {e}")
        return MovieFilters()
    except Exception as e:
        st.error(f"Error parsing filters: {e}")
        return MovieFilters()

# Enhanced search function
def search_movies(query, top_k=5):
    # Get conversation history as string
    history_str = "\n".join([f"Q: {item['query']}\nA: {item['response']}" 
                           for item in st.session_state.conversation_history[-3:]])
    
    try:
        # Extract filters using conversation context
        filters_response = filter_chain.run(query=query, conversation_history=history_str)
        filters = parse_filters(filters_response)
        
        # Display the filters in a pretty format
        st.markdown("**Extracted Filters:**")
        st.json(filters.model_dump())  # Using model_dump() instead of json()
        
    except Exception as e:
        st.warning(f"Could not parse filters: {e}")
        filters = MovieFilters()

    # Perform similarity search
    search_kwargs = {"k": top_k}
    results = vectorstore.similarity_search(query, **search_kwargs)
    st.session_state.last_results = results
    
    st.markdown("### 🎥 Top Movie Recommendations")
    for i, res in enumerate(results):
        st.markdown(f"**{i+1}.** {res.page_content}")
        if hasattr(res, 'metadata') and res.metadata:
            st.markdown(f"*Metadata: {res.metadata}*")
        st.markdown("---")
    
    return results

# Display conversation history
if st.session_state.conversation_history:
    st.sidebar.markdown("### 💬 Recent Conversations")
    for i, conv in enumerate(st.session_state.conversation_history[-3:]):
        with st.sidebar.expander(f"Query {len(st.session_state.conversation_history) - 2 + i}"):
            st.markdown(f"**Q:** {conv['query']}")
            st.markdown(f"**A:** {conv['response'][:100]}...")

# Main search functionality
if st.button("🔍 Search") or query:
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Searching for movies..."):
            results = search_movies(query)
            
            # Store in conversation history
            response_summary = f"Found {len(results)} movie recommendations"
            st.session_state.conversation_history.append({
                'query': query,
                'response': response_summary,
                'results': results
            })

# Clear conversation history
if st.sidebar.button("🗑️ Clear History"):
    st.session_state.conversation_history = []
    st.session_state.last_results = []
    st.session_state.conversation_memory.clear()
    st.rerun()