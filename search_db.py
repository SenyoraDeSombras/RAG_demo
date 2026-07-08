import os
import json
import argparse
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

# Initialize ChromaDB client and connection to the collection
CHROMA_DB_PATH = "./scientific_db"
COLLECTION_NAME = "articles"

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Configure embedding function (matching make_db.py configuration)
gemma_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_base="http://localhost:8080/v1",
    model_name="gemma4",
    api_key="no-key-required"
)

# Get the collection
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=gemma_ef
)

# Initialize OpenAI client to interact with a local LLM server (e.g., llama.cpp)
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="no-key-required"
)

def query_database(search_term: str) -> str:
    """
    Queries the ChromaDB collection for documents relevant to the search term.
    
    Args:
        search_term: The text to search for in the vector database.
        
    Returns:
        A formatted string containing the most relevant document chunks and their metadata.
    """
    # Query for more results than strictly needed to allow for duplicate filtering
    results = collection.query(query_texts=[search_term], n_results=7)
    
    if not results['documents'] or len(results['documents'][0]) == 0:
        return "No relevant data found in the local documents."
        
    seen_contents = set()
    formatted_results = []
    
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        # Remove extra whitespace for comparison
        clean_doc = doc.strip()
        
        # Skip if we've already included this exact content
        if clean_doc in seen_contents:
            continue
            
        seen_contents.add(clean_doc)
        
        # Get source file and header structure for context
        source_file = meta.get('source', 'Unknown file')
        headers = [meta[h] for h in ["Header_1", "Header_2", "Header_3"] if h in meta]
        context_path = " > ".join(headers) if headers else "Main Text"
        
        formatted_results.append(f"[File: {source_file} | Section: {context_path}]\n{doc}")
        
        # Limit to top unique fragments for context efficiency
        if len(formatted_results) >= 7:
            break
            
    if not formatted_results:
        return "No relevant information found."
        
    return "\n\n---\n\n".join(formatted_results)

def ask_gemma(user_question: str, max_turns: int = 5):
    """
    Runs a reasoning loop with the LLM, allowing it to query the database multiple times 
    if its first answer is insufficient.
    
    Args:
        user_question: The original question from the user.
        max_turns: Maximum number of tool-calling iterations allowed.
        
    Returns:
        The final answer provided by the LLM.
    """
    # Define the tool for the LLM to use
    tools = [
        {
            "type": "function",
            "function": {
                "name": "query_database",
                "description": "Use this tool to search for precise physical-chemical parameters, numbers, temperatures, and concentrations in the scientific papers database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_term": {
                            "type": "string",
                            "description": "Keywords or entities to search for."
                        }
                    },
                    "required": ["search_term"]
                }
            }
        }
    ]

    messages = [
        {
            "role": "system", 
            "content": (
                "You are a strict scientific assistant. If your first search doesn't provide enough information, "
                "you MUST try to refine your search query (e.g., using synonyms or broader terms) to find the "
                "required facts. Do not give up immediately."
            )
        },
        {"role": "user", "content": user_question}
    ]

    # Reasoning Loop
    for turn in range(max_turns):
        print(f"\n[Reasoning Step {turn + 1}] Sending request to LLM...")
        
        response = client.chat.completions.create(
            model="gemma-4",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # If the model doesn't call any tools, it has provided a final answer
        if not tool_calls:
            print("[LLM formulated final answer]")
            return response_message.content

        # If the model called tools, save the intent to history
        messages.append(response_message)
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            search_term = function_args.get("search_term")
            
            if function_name == "query_database":
                print(f" -> LLM searching for: '{search_term}'")
                
                db_result = query_database(search_term=search_term)
                print(f" <- Database returned: {db_result[:100]}..." if len(db_result) > 100 else f" <- Database returned: {db_result}")
                
                # Append the tool result to the conversation history
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": db_result
                })

    return "The LLM reached the maximum number of search attempts without finding sufficient data."

def main():
    parser = argparse.ArgumentParser(description="Query scientific information using a RAG system.")
    parser.add_argument("question", type=str, help="The question you want to ask the scientific assistant.")
    
    args = parser.parse_args()
    
    print(f"Question: {args.question}")
    answer = ask_gemma(args.question)
    print(f"\nLLM Answer: {answer}")

if __name__ == "__main__":
    main()
