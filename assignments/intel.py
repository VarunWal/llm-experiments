import os
import json
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_mistralai import ChatMistralAI
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Set up Streamlit UI
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Natural Language Movie Recommender")

# Initialize session state for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'last_results' not in st.session_state:
    st.session_state.last_results = []
if 'conversation_memory' not in st.session_state:
    st.session_state.conversation_memory = ConversationBufferMemory()

# Sidebar config
# st.sidebar.header("🔧 Configuration")
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
    template="""Based on the user's query and conversation history, extract relevant movie filters. also dont include json tag like ```json ``` just return a json 

Previous conversation:
{conversation_history}

Current query: {query}
also dont include json tag like ```json ```
Extract filters as valid JSON with possible keys: genre, date_range, actors, plot details
also dont include json tag like ```json ```
Return only valid JSON, no other text. also dont include json tag like ```json ```
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

# Enhanced search function
def search_movies(query, top_k=5):
    # Get conversation history as string
    history_str = "\n".join([f"Q: {item['query']}\nA: {item['response']}" 
                            for item in st.session_state.conversation_history[-3:]])  # Last 3 exchanges
    
    try:
        # Extract filters using conversation context
        filters_response = filter_chain.run(query=query, conversation_history=history_str)
        filters = json.loads(filters_response)
        
        st.markdown("**Extracted Filters:**")
        st.code(json.dumps(filters, indent=2), language="json")
        
    except (json.JSONDecodeError, Exception) as e:
        st.warning(f"Could not parse filters: {e}")
        filters = {}

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