import os
import glob
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# Initialize ChromaDB client
# The persistent storage is located in './scientific_db'
chroma_client = chromadb.PersistentClient(path="./scientific_db")

# Configure the embedding function to use a local OpenAI-compatible server (e.g., llama.cpp)
gemma_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_base="http://localhost:8080/v1",
    model_name="gemma4",
    api_key="no-key-required"
)

# Create or get the existing 'articles' collection
collection = chroma_client.get_or_create_collection(
    name="articles", 
    embedding_function=gemma_ef
)

# Configuration for text splitting
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)

def ingest_markdown_with_langchain(folder_path: str):
    """
    Scans a folder for Markdown files, splits them into chunks while preserving 
    header structure, and adds them to the ChromaDB collection.
    
    Args:
        folder_path: Path to the directory containing Markdown files.
    """
    print(f"Scanning folder: {folder_path}")
    
    # Find all .md files recursively
    search_pattern = os.path.join(folder_path, "**", "*.md")
    md_files = glob.glob(search_pattern, recursive=True)
    
    if not md_files:
        print(f"No Markdown files found in {folder_path}")
        return

    # Define headers to split the Markdown document on
    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    chunk_counter = 0
    
    for file_path in md_files:
        file_name = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            # Step 1: Split the text by Markdown header structure to capture hierarchy in metadata
            md_header_splits = markdown_splitter.split_text(text)
            
            all_chunks = []
            all_metadatas = []
            all_ids = []
            
            # Step 2: Further split the header-aware documents into smaller chunks
            txt_splits = text_splitter.split_documents(md_header_splits)
            
            for split in txt_splits:
                # Copy metadata and add the source filename
                metadata = split.metadata.copy()
                metadata["source"] = file_name
                
                all_chunks.append(split.page_content)
                all_metadatas.append(metadata)
                all_ids.append(f"id_{file_name}_{chunk_counter}")
                chunk_counter += 1
                
            if all_chunks:
                collection.add(
                    documents=all_chunks,
                    metadatas=all_metadatas,
                    ids=all_ids
                )
                print(f" Successfully added: {file_name} ({len(all_chunks)} chunks)")
                
        except Exception as e:
            print(f" Error processing {file_name}: {e}")

if __name__ == "__main__":
    # Default ingestion directory
    target_folder = "./my_papers"
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"Created directory: {target_folder}. Please place your .md files there.")
    else:
        ingest_markdown_with_langchain(target_folder)
