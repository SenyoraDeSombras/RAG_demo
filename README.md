# Scientific RAG Assistant

A Retrieval-Augmented Generation (RAG) system designed to query scientific information from Markdown documents. This system uses a vector database for efficient similarity search and employs a "Reasoning Loop" pattern, allowing a Large Language Model (LLM) to autonomously decide to query the database multiple times to refine its answers.

## 🌟 Features

- **Hierarchical Markdown Parsing:** Preserves document structure (headers) in metadata for better context awareness.
- **Reasoning Loop:** The LLM can perform iterative searches if the first attempt doesn't yield sufficient results.
- **Local & Private:** Designed to work with local LLM servers (like `llama.cpp`), ensuring data privacy.
- **Efficient Search:** Uses ChromaDB for high-performance vector similarity search.

## 🚀 Technology Stack

- **Language:** Python
- **Vector Database:** [ChromaDB](https://www.trychroma.com/)
- **Text Processing:** [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- **LLM Interface:** OpenAI-compatible API (e.g., `llama.cpp`, `Ollama`, or OpenAI)

## 🛠️ Prerequisites

- Python 3.9 or higher
- A running OpenAI-compatible LLM server at `http://localhost:8080/v1` (e.g., `llama.cpp`)

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage

### 1. Data Ingestion

Place your scientific papers in Markdown (`.md`) format into a folder named `my_papers`. If the folder does not exist, the system will create it for you.

Run the ingestion script:
```bash
python make_db.py
```
This will process the files and create a persistent vector database in the `./scientific_db` directory.

### 2. Querying

Ask questions about your documents directly from the command line:
```bash
python search_db.py "What is the temperature of the reaction mentioned in the documents?"
```

The system will enter a reasoning loop, searching the database as needed until it finds a satisfactory answer or reaches the maximum number of attempts.

## 📂 Project Structure

- `make_db.py`: Script for data ingestion and vector database population.
- `search_db.py`: The main query interface with an LLM reasoning loop.
- `requirements.txt`: List of required Python packages.
- `README.md`: Project documentation.
- `scientific_db/`: (Generated) Persistent ChromaDB storage.
- `my_papers/`: (User-provided) Directory for Markdown files.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📜 License

This project is licensed under the GPL v. 3.0 License.
