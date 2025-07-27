import os
from langchain_mistralai import ChatMistralAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma

# Initialize Mistral LLM
llm = ChatMistralAI(model="mistral-small-latest", temperature=0.3)

# Prompt for extracting filters
prompt_template = PromptTemplate(
    input_variables=["query"],
    template="""Extract relevant filters like genre, date range, actor, or crew from this movie query:
Query: {query}
Respond as JSON with keys like genre, date_range, actor, crew.
""")

filter_chain = LLMChain(llm=llm, prompt=prompt_template)

# Load existing vector store (assumes embeddings and Chroma DB are already created)
vectorstore = Chroma(persist_directory="./chroma_store")

# Inference function
def search_movies(query, top_k=5):
    filter_info = filter_chain.run(query)
    print("Extracted Filters:", filter_info)
    print("\nTop Recommendations:\n")

    # Use raw query for semantic search (can be improved with structured filters)
    results = vectorstore.similarity_search(query, k=top_k)
    for i, res in enumerate(results):
        print(f"{i+1}. {res.page_content}\n{'-'*80}")

# Example Usage
if __name__ == "__main__":
    user_query = "Looking for a crime thriller featuring Leonardo DiCaprio"
    search_movies(user_query)
