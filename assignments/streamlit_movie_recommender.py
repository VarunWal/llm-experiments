import os
import json
import streamlit as st
from langchain.chains import ConversationChain
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

# Followup question generation prompt
followup_prompt = PromptTemplate(
    input_variables=["query", "results", "conversation_history"],
    template="""Based on the user's query, search results, and conversation history, generate 3-4 helpful followup questions to refine their movie search.

User query: {query}
Search results: {results}
Previous conversation: {conversation_history}

Generate followup questions that help the user:
1. Narrow down their preferences
2. Explore related movies
3. Get more specific recommendations
4. Learn more about the recommended movies

Return 3-4 questions as a JSON array of strings.
"""
)

filter_chain = LLMChain(llm=llm, prompt=filter_prompt)
followup_chain = LLMChain(llm=llm, prompt=followup_prompt)

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
    # You can enhance this to use the extracted filters for metadata filtering
    search_kwargs = {"k": top_k}
    if filters:
        # Add metadata filtering if your Chroma store supports it
        # search_kwargs["filter"] = filters
        pass
    
    results = vectorstore.similarity_search(query, **search_kwargs)
    st.session_state.last_results = results
    
    st.markdown("### 🎥 Top Movie Recommendations")
    for i, res in enumerate(results):
        st.markdown(f"**{i+1}.** {res.page_content}")
        if hasattr(res, 'metadata') and res.metadata:
            st.markdown(f"*Metadata: {res.metadata}*")
        st.markdown("---")
    
    return results

# Generate followup questions
def generate_followup_questions(query, results):
    try:
        history_str = "\n".join([f"Q: {item['query']}\nA: {item['response']}" 
                                for item in st.session_state.conversation_history[-2:]])
        
        results_str = "\n".join([f"- {res.page_content}" for res in results[:3]])
        
        followup_response = followup_chain.run(
            query=query, 
            results=results_str, 
            conversation_history=history_str
        )
        
        followup_questions = json.loads(followup_response)
        return followup_questions
        
    except (json.JSONDecodeError, Exception) as e:
        st.warning(f"Could not generate followup questions: {e}")
        return [
            "Would you like movies from a specific decade?",
            "Are you interested in a particular genre?",
            "Would you like similar movies to any of these recommendations?",
            "Do you have a preferred movie rating or length?"
        ]

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
            
            # Generate and display followup questions
            # st.markdown("### 🤔 Followup Questions")
            # followup_questions = generate_followup_questions(query, results)
            
            # Display followup questions as clickable buttons
            # col1, col2 = st.columns(2)
            # for i, question in enumerate(followup_questions):
            #     if i % 2 == 0:
            #         with col1:
            #             if st.button(f"❓ {question}", key=f"followup_{i}"):
            #                 st.session_state.followup_query = question
            #                 st.rerun()
            #     else:
            #         with col2:
            #             if st.button(f"❓ {question}", key=f"followup_{i}"):
            #                 st.session_state.followup_query = question
            #                 st.rerun()

# Handle followup question clicks
if hasattr(st.session_state, 'followup_query'):
    st.info(f"Following up on: {st.session_state.followup_query}")
    with st.spinner("Searching based on followup..."):
        results = search_movies(st.session_state.followup_query)
        
        # Update conversation history
        response_summary = f"Followup search found {len(results)} recommendations"
        st.session_state.conversation_history.append({
            'query': st.session_state.followup_query,
            'response': response_summary,
            'results': results
        })
    
    # Clear the followup query
    del st.session_state.followup_query

# Add refinement options
# st.markdown("### 🎛️ Quick Refinements")
# if st.session_state.last_results:
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         if st.button("🔄 Find Similar Movies"):
#             if st.session_state.last_results:
#                 similar_query = f"Movies similar to {st.session_state.last_results[0].page_content}"
#                 search_movies(similar_query)
    
#     with col2:
#         if st.button("📅 Same Decade"):
#             decade_query = "Movies from the same time period as the recommendations"
#             search_movies(decade_query)
    
#     with col3:
#         if st.button("🎭 Same Genre"):
#             genre_query = "Movies in the same genre as the recommendations"
#             search_movies(genre_query)

# Clear conversation history
if st.sidebar.button("🗑️ Clear History"):
    st.session_state.conversation_history = []
    st.session_state.last_results = []
    st.session_state.conversation_memory.clear()
    st.rerun()